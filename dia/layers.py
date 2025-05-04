import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor
from torch.nn import RMSNorm
from typing import Optional # Import Optional

from .config import DiaConfig
from .state import DecoderInferenceState, EncoderInferenceState, KVCache


def _normalize_axes(axes: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


class DenseGeneral(nn.Module):
    """
    PyTorch equivalent of flax.linen.DenseGeneral with shapes defined at init.

    Stores weights (`kernel`) in the same layout as Jax and uses torch.tensordot
    for the generalized matrix multiplication. Weight/bias shapes are calculated
    and parameters created during initialization based on config.
    `load_weights` validates shapes and copies data.

    Attributes:
        axis (Tuple[int, ...]): Input axis or axes to contract.
        in_shapes (Tuple[int, ...]): Sizes of the input dimensions specified by `axis`.
        out_features (Tuple[int, ...]): Shape of the output features (non-contracted dims).
        use_bias (bool): Whether to add a bias term.
        weight (nn.Parameter): The kernel parameter.
        bias (Optional[nn.Parameter]): The bias parameter (if use_bias=True).
    """

    def __init__(
        self,
        in_shapes: tuple[int, ...],
        out_features: tuple[int, ...],
        axis: tuple[int, ...] = (-1,),
        weight_dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_shapes = in_shapes
        self.out_features = out_features
        self.axis = axis
        self.kernel_shape = self.in_shapes + self.out_features

        factory_kwargs = {"device": device, "dtype": weight_dtype}
        # Use 'weight' instead of 'kernel' to match PyTorch conventions for state_dict loading
        self.weight = nn.Parameter(torch.empty(self.kernel_shape, **factory_kwargs))
        # Note: Bias is not implemented here, matching the original code's apparent lack of bias in DenseGeneral usage.
        # If bias is needed, it should be added here.

    def forward(self, inputs: Tensor) -> Tensor:
        norm_axis = _normalize_axes(self.axis, inputs.ndim)
        kernel_contract_axes = tuple(range(len(norm_axis)))

        # Ensure inputs are compatible dtype for tensordot if weight has specific dtype
        input_dtype = inputs.dtype
        output = torch.tensordot(
            inputs.to(self.weight.dtype),
            self.weight,
            dims=(norm_axis, kernel_contract_axes),
        ).to(input_dtype) # Cast back to original input dtype
        return output


class MlpBlock(nn.Module):
    """MLP block using DenseGeneral."""

    def __init__(self, embed_dim: int, intermediate_dim: int, compute_dtype: torch.dtype):
        super().__init__()
        self.dtype = compute_dtype

        # Fused gate and up projection
        self.wi_fused = DenseGeneral(
            in_shapes=(embed_dim,),
            out_features=(2, intermediate_dim), # Output features: (gate, up)
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

        # Down projection
        self.wo = DenseGeneral(
            in_shapes=(intermediate_dim,),
            out_features=(embed_dim,),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Project and split gate/up
        fused_x = self.wi_fused(x) # Shape: [..., 2, intermediate_dim]
        gate = fused_x[..., 0, :] # Shape: [..., intermediate_dim]
        up = fused_x[..., 1, :]   # Shape: [..., intermediate_dim]

        # Apply activation (SiLU) and multiply
        # Ensure intermediate computation uses appropriate dtype if needed
        hidden = torch.mul(F.silu(gate.to(torch.float32)).to(self.dtype), up)

        # Project down
        output = self.wo(hidden)
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation in PyTorch."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.compute_dtype = dtype # Store the intended compute dtype

        half_embedding_dim = embedding_dims // 2
        # Calculate frequencies (inverse timescales)
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        inv_freq = 1.0 / (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction)

        # Register buffer expects a tensor, not just a float
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)


    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        # inputs shape: [B, ..., N, D] or [B, ..., D] where D = embedding_dims
        # position shape: [B, ..., T] or broadcastable
        # Ensure position is broadcastable to input shape for frequency calculation
        # Typically position is [B, T] or [B, 1] during decoding
        # We need freqs shape compatible with input for elementwise multiplication

        # Calculate theta = position * inv_freq
        # Position needs shape like [B, T, 1] or [B, 1, 1] to broadcast with inv_freq [H/2]
        pos_expanded = position.unsqueeze(-1) # Shape [B, T, 1] or [B, 1, 1]
        freqs = pos_expanded * self.inv_freq.to(pos_expanded.device) # Shape [B, T, H/2] or [B, 1, H/2]

        # Expand freqs to match the input dimensions for sin/cos calculation
        # Input shape is likely [B, N, T, H] or [B, T, N, H] after projection/transpose
        # Let's assume input is [..., D] and freqs is [..., D/2] after repeat_interleave
        freqs = torch.cat((freqs, freqs), dim=-1) # Shape [..., D]

        # Reshape freqs to match input dimensions if necessary
        # Example: If input is [B, N, T, H], freqs needs to be [B, 1, T, H] or similar
        # This depends heavily on the Attention implementation details.
        # Assuming freqs shape is compatible for now.
        # A common pattern: freqs shape [T, H] or [B, T, H] -> apply to input [B, N, T, H]

        # Calculate sin and cos
        # Ensure calculations happen in float32 for stability
        sin = torch.sin(freqs.to(torch.float32))
        cos = torch.cos(freqs.to(torch.float32))

        # Split input into two halves
        x1, x2 = torch.chunk(inputs.to(torch.float32), 2, dim=-1)

        # Apply rotation: [-x2*sin + x1*cos, x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Concatenate and cast back to original compute dtype
        output = torch.cat((rotated_x1, rotated_x2), dim=-1)
        return output.to(self.compute_dtype)


class Attention(nn.Module):
    """Attention using DenseGeneral."""

    def __init__(
        self,
        config: DiaConfig,
        q_embed_dim: int,
        kv_embed_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compute_dtype: torch.dtype,
        is_cross_attn: bool = False,
        out_embed_dim: int | None = None,
    ):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.is_cross_attn = is_cross_attn
        self.output_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.projected_query_dim = num_query_heads * head_dim
        if num_query_heads % num_kv_heads != 0:
            raise ValueError(f"num_query_heads ({num_query_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        self.num_gqa_groups = num_query_heads // num_kv_heads
        self.compute_dtype = compute_dtype # Store compute dtype

        # --- Projection Layers using DenseGeneral ---
        self.q_proj = DenseGeneral(
            in_shapes=(q_embed_dim,),
            out_features=(num_query_heads, head_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )
        self.k_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )
        self.v_proj = DenseGeneral(
            in_shapes=(kv_embed_dim,),
            out_features=(num_kv_heads, head_dim),
            axis=(-1,),
            weight_dtype=compute_dtype,
        )
        self.o_proj = DenseGeneral(
            in_shapes=(num_query_heads, head_dim),
            out_features=(self.output_dim,),
            axis=(-2, -1), # Contract head and head_dim axes
            weight_dtype=compute_dtype,
        )

        # --- Rotary Embedding ---
        # RoPE applied after projection, before attention calculation
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=self.head_dim,
            min_timescale=config.model.rope_min_timescale,
            max_timescale=config.model.rope_max_timescale,
            dtype=compute_dtype, # Pass compute_dtype
        )

    def forward(
        self,
        Xq: torch.Tensor,  # Query input (B, Tq, Dq) Tq=1 in AR generation
        Xkv: torch.Tensor, # Key/Value input (B, Tkv, Dkv) Tkv=1 in AR self-attn
        q_positions: torch.Tensor,  # Query positions (B, Tq)
        kv_positions: torch.Tensor | None = None,  # Key/Value positions (B, Tkv). If None, uses q_positions (for self-attn).
        attn_mask: torch.Tensor | None = None,  # Attention mask (B, 1, Tq, Tkv_eff). Tkv_eff includes cache.
        cache: KVCache | None = None,  # KV cache object for autoregressive decoding.
        prefill: bool = False, # Flag for prefill mode (populating cache initially)
        is_causal: bool = False, # Flag for enabling causal masking in SDPA (used in decoder self-attn prefill)
    ) -> torch.Tensor: # Return only the output tensor
        """
        Performs attention calculation with optional KV caching and GQA/MQA.

        Args:
            Xq: Query tensor (B, Tq, Dq). Tq=1 during single-step decoding.
            Xkv: Key/Value source tensor (B, Tkv, Dkv). Tkv=1 during single-step decoding for self-attn.
            q_positions: Positions for queries (B, Tq).
            kv_positions: Positions for keys/values (B, Tkv). If None, uses q_positions.
            attn_mask: Attention mask (B, 1, Tq, Tkv_eff). Applied *before* softmax in SDPA.
            cache: KVCache object. If provided, enables KV caching.
            prefill: If True, indicates the prefill phase for the KV cache.
            is_causal: If True, apply causal mask within SDPA (useful for self-attn prefill).

        Returns:
            output: The attention output tensor (B, Tq, output_dim).
        """
        if kv_positions is None:
            kv_positions = q_positions # Use query positions for K/V if not provided (self-attention)
        original_dtype = Xq.dtype
        B, Tq, _ = Xq.shape
        _, Tkv, _ = Xkv.shape

        # 1. Project Q, K, V
        # Use compute_dtype for projections if different from input
        q = self.q_proj(Xq.to(self.compute_dtype)) # (B, Tq, Nq, H)
        k = self.k_proj(Xkv.to(self.compute_dtype)) # (B, Tkv, Nkv, H)
        v = self.v_proj(Xkv.to(self.compute_dtype)) # (B, Tkv, Nkv, H)

        # 2. Apply Rotary Embeddings
        q = self.rotary_emb(q, position=q_positions)
        k = self.rotary_emb(k, position=kv_positions)

        # 3. Handle KV Cache
        if cache is not None:
            # Cross-attention: K/V comes entirely from cache (precomputed from encoder)
            if self.is_cross_attn:
                # During inference, cross-attn K/V are static and precomputed
                # The 'cache' object here would hold the precomputed enc_k, enc_v
                attn_k, attn_v = cache.k, cache.v # Shapes [B, Nkv, S, H]
                # Tkv_eff (effective key/value sequence length) is the encoder output length S
            # Self-attention: Update cache with current K/V
            else:
                # Transpose K/V for cache: (B, Tkv, Nkv, H) -> (B, Nkv, Tkv, H)
                k_cache_format = k.transpose(1, 2)
                v_cache_format = v.transpose(1, 2)

                if prefill:
                    # Store the initial K/V sequence in the cache
                    attn_k, attn_v = cache.prefill(k_cache_format, v_cache_format)
                    # Tkv_eff is Tkv (the prefill length)
                else:
                    # Append current K/V step to the cache
                    attn_k, attn_v = cache.update(k_cache_format, v_cache_format)
                    # Tkv_eff is cache.current_idx (total length including current step)
        else:
            # No cache used (e.g., during encoder self-attention or training)
            # Transpose K/V for attention: (B, Tkv, Nkv, H) -> (B, Nkv, Tkv, H)
            attn_k = k.transpose(1, 2)
            attn_v = v.transpose(1, 2)
            # Tkv_eff is Tkv

        # 4. Prepare Q for Attention
        # Transpose Q: (B, Tq, Nq, H) -> (B, Nq, Tq, H)
        attn_q = q.transpose(1, 2)

        # 5. Repeat K/V for Grouped Query Attention (GQA/MQA) if needed
        if self.num_gqa_groups > 1:
            # attn_k shape: [B, Nkv, Tkv_eff, H]
            # attn_v shape: [B, Nkv, Tkv_eff, H]
            # Repeat Nkv heads to match Nq heads
            attn_k = attn_k.repeat_interleave(self.num_gqa_groups, dim=1) # Shape [B, Nq, Tkv_eff, H]
            attn_v = attn_v.repeat_interleave(self.num_gqa_groups, dim=1) # Shape [B, Nq, Tkv_eff, H]


        # 6. Scaled Dot-Product Attention (SDPA)
        # attn_q: (B, Nq, Tq, H)
        # attn_k: (B, Nq, Tkv_eff, H)
        # attn_v: (B, Nq, Tkv_eff, H)
        # attn_mask: (B, 1, Tq, Tkv_eff) - Broadcasts over Nq heads
        # is_causal: bool - Applies causal mask if True (only for self-attn prefill)
        attn_output = F.scaled_dot_product_attention(
            attn_q,
            attn_k,
            attn_v,
            attn_mask=attn_mask,
            is_causal=is_causal and not self.is_cross_attn, # Only apply causal mask in self-attention prefill
            dropout_p=0.0 # No dropout during inference/typically
            # scale=None # Defaults to 1/sqrt(head_dim)
        ) # Output shape: (B, Nq, Tq, H)

        # 7. Reshape and Project Output
        # Transpose back: (B, Nq, Tq, H) -> (B, Tq, Nq, H)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Project: (B, Tq, Nq, H) -> (B, Tq, output_dim)
        output = self.o_proj(attn_output)

        # Cast back to original input dtype
        return output.to(original_dtype)


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        embed_dim = enc_config.n_embd
        self.compute_dtype = compute_dtype

        self.pre_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            # RMSNorm often kept in float32 for stability
            # dtype=torch.float32,
        )
        self.self_attention = Attention(
            config,
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            num_query_heads=enc_config.n_head,
            num_kv_heads=enc_config.n_head, # MHA in encoder
            head_dim=enc_config.head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=False,
            out_embed_dim=embed_dim,
        )
        # Post-attention norm (often called pre-MLP norm)
        self.post_sa_norm = RMSNorm(
            embed_dim,
            eps=model_config.normalization_layer_epsilon,
            # dtype=torch.float32,
        )
        self.mlp = MlpBlock(embed_dim=embed_dim, intermediate_dim=enc_config.n_hidden, compute_dtype=compute_dtype)

    def forward(
        self,
        x: torch.Tensor, # Input tensor (B, T, D)
        state: EncoderInferenceState, # Contains positions and mask
    ) -> torch.Tensor:
        # --- Self-Attention Block ---
        residual = x
        # Use float32 for norm calculation for stability
        x_norm = self.pre_sa_norm(x.to(torch.float32)).to(self.compute_dtype)

        # Pass state.positions for RoPE, state.attn_mask for masking
        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm,
            q_positions=state.positions,
            kv_positions=state.positions, # Self-attention
            attn_mask=state.attn_mask, # Precomputed mask
            cache=None, # No KV cache in encoder forward pass
            is_causal=False # Encoder self-attention is not causal
        )
        # Add residual (skip connection)
        x = residual + sa_out

        # --- MLP Block ---
        residual = x
        # Use float32 for norm calculation
        x_norm = self.post_sa_norm(x.to(torch.float32)).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        # Add residual
        x = residual + mlp_out

        return x


class Encoder(nn.Module):
    """Transformer Encoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        enc_config = config.model.encoder
        self.compute_dtype = compute_dtype

        self.embedding = nn.Embedding(
            model_config.src_vocab_size,
            enc_config.n_embd,
            # Padding index is not explicitly used here, handle via mask
            # dtype=compute_dtype, # Embeddings often kept in float32
        )
        self.layers = nn.ModuleList([EncoderLayer(config, compute_dtype) for _ in range(enc_config.n_layer)])
        # Final norm after all layers
        self.norm = RMSNorm(
            enc_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            # dtype=torch.float32,
        )
        # Dropout is typically applied during training, not inference
        # self.dropout = nn.Dropout(model_config.dropout)

    def forward(
        self,
        x_ids: torch.Tensor, # Input token IDs (B, T)
        state: EncoderInferenceState, # Contains positions and mask
    ) -> torch.Tensor:
        # 1. Embeddings
        # Use float32 for embeddings? Or compute_dtype? Let's try compute_dtype.
        x = self.embedding(x_ids).to(self.compute_dtype)
        # Apply dropout if training: x = self.dropout(x)

        # 2. Transformer Layers
        for layer in self.layers:
            x = layer(x, state)

        # 3. Final Normalization
        # Use float32 for norm calculation
        x = self.norm(x.to(torch.float32)).to(self.compute_dtype)
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        enc_config = config.model.encoder
        dec_embed_dim = dec_config.n_embd
        enc_embed_dim = enc_config.n_embd # Encoder output dim
        self.compute_dtype = compute_dtype

        # --- Norms ---
        self.pre_sa_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            # dtype=torch.float32,
        )
        self.pre_ca_norm = RMSNorm(
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            # dtype=torch.float32,
        )
        self.pre_mlp_norm = RMSNorm( # Renamed from post_ca_norm for clarity
            dec_embed_dim,
            eps=model_config.normalization_layer_epsilon,
            # dtype=torch.float32,
        )

        # --- Self-Attention (GQA) ---
        # Causal masking is handled by SDPA `is_causal=True` during prefill
        # or implicitly by the KV cache structure during step-by-step decoding.
        self.self_attention = Attention(
            config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=dec_embed_dim,
            num_query_heads=dec_config.gqa_query_heads,
            num_kv_heads=dec_config.kv_heads, # GQA/MQA
            head_dim=dec_config.gqa_head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=False,
            out_embed_dim=dec_embed_dim,
        )

        # --- Cross-Attention (MHA) ---
        self.cross_attention = Attention(
            config=config,
            q_embed_dim=dec_embed_dim,
            kv_embed_dim=enc_embed_dim,  # K/V comes from encoder output
            num_query_heads=dec_config.cross_query_heads,
            num_kv_heads=dec_config.cross_query_heads, # MHA
            head_dim=dec_config.cross_head_dim,
            compute_dtype=compute_dtype,
            is_cross_attn=True,
            out_embed_dim=dec_embed_dim,
        )

        # --- MLP ---
        self.mlp = MlpBlock(
            embed_dim=dec_embed_dim,
            intermediate_dim=dec_config.n_hidden,
            compute_dtype=compute_dtype,
        )

    def forward(
        self,
        x: torch.Tensor, # Input tensor (B, Tq, D) Tq=1 in AR generation
        state: DecoderInferenceState, # Contains enc_out, positions, masks, caches
        self_attn_cache: KVCache | None = None, # Passed explicitly for clarity
        cross_attn_cache: KVCache | None = None, # Passed explicitly
        prefill: bool = False, # Indicates prefill mode
    ) -> torch.Tensor:
        # --- Self-Attention Block ---
        residual = x
        # Use float32 for norm
        x_norm = self.pre_sa_norm(x.to(torch.float32)).to(self.compute_dtype)

        # Self-attention with KV cache
        sa_out = self.self_attention(
            Xq=x_norm,
            Xkv=x_norm, # K/V source is also x_norm for self-attn
            q_positions=state.dec_positions, # Current decoder positions
            kv_positions=state.dec_positions, # Self-attn uses same positions
            attn_mask=None, # Causal mask handled by SDPA or cache structure
            cache=self_attn_cache,
            prefill=prefill,
            is_causal=prefill, # Apply explicit causal mask only during prefill
        )
        # Add residual
        x = residual + sa_out

        # --- Cross-Attention Block ---
        residual = x
        # Use float32 for norm
        x_norm = self.pre_ca_norm(x.to(torch.float32)).to(self.compute_dtype)

        # Cross-attention with precomputed encoder K/V cache
        ca_out = self.cross_attention(
            Xq=x_norm,
            Xkv=state.enc_out, # K/V source is encoder output
            q_positions=state.dec_positions, # Query positions are decoder positions
            kv_positions=state.enc_positions, # K/V positions are encoder positions
            attn_mask=state.dec_cross_attn_mask, # Use precomputed cross-attn mask
            cache=cross_attn_cache, # Pass the precomputed cross-attn K/V
            prefill=False, # Cross-attn cache is static, no prefill/update needed here
            is_causal=False, # Cross-attention is never causal
        )
        # Add residual
        x = residual + ca_out

        # --- MLP Block ---
        residual = x
        # Use float32 for norm
        x_norm = self.pre_mlp_norm(x.to(torch.float32)).to(self.compute_dtype)
        mlp_out = self.mlp(x_norm)
        # Add residual
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    """Transformer Decoder Stack using DenseGeneral."""

    def __init__(self, config: DiaConfig, compute_dtype: torch.dtype):
        super().__init__()
        self.config = config
        model_config = config.model
        dec_config = config.model.decoder
        data_config = config.data
        self.num_channels = data_config.channels
        self.num_layers = dec_config.n_layer
        self.compute_dtype = compute_dtype # Store compute dtype

        # Embeddings for each audio channel/codebook
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    model_config.tgt_vocab_size,
                    dec_config.n_embd,
                    # dtype=compute_dtype # Embeddings often kept in float32
                    )
                for _ in range(self.num_channels)
            ]
        )
        # Decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(config=config, compute_dtype=compute_dtype) for _ in range(self.num_layers)]
        )
        # Final normalization
        self.norm = RMSNorm(
            dec_config.n_embd,
            eps=model_config.normalization_layer_epsilon,
            # dtype=torch.float32,
        )
        # Output projection layer (logits head)
        self.logits_dense = DenseGeneral(
            in_shapes=(dec_config.n_embd,),
            # Output shape: (num_channels, vocab_size)
            out_features=(self.num_channels, model_config.tgt_vocab_size),
            axis=(-1,), # Contract last dimension (embedding dim)
            weight_dtype=compute_dtype,
        )
        # Dropout (applied during training)
        # self.dropout = nn.Dropout(model_config.dropout)

    def precompute_cross_attn_cache(
        self,
        enc_out: torch.Tensor,  # Encoder output (B, S, E)
        enc_positions: torch.Tensor,  # Encoder positions (B, S)
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer
        from the encoder output. These are static during decoding.
        """
        per_layer_kv_cache: list[KVCache] = []

        # Ensure model is in eval mode for this precomputation if called separately
        is_training = self.training
        self.eval()

        with torch.no_grad(): # No need for gradients here
            for layer in self.layers:
                cross_attn_module = layer.cross_attention
                # Project encoder output to K, V for this layer
                # Use compute_dtype for projection
                k_proj = cross_attn_module.k_proj(enc_out.to(self.compute_dtype)) # (B, S, Nkv, H)
                v_proj = cross_attn_module.v_proj(enc_out.to(self.compute_dtype)) # (B, S, Nkv, H)

                # Apply RoPE to keys based on encoder positions
                k_proj = cross_attn_module.rotary_emb(k_proj, position=enc_positions)

                # Transpose for cache format: (B, Nkv, S, H)
                k = k_proj.transpose(1, 2)
                v = v_proj.transpose(1, 2)

                # Create KVCache object holding these static K/V tensors
                per_layer_kv_cache.append(KVCache.from_kv(k, v))

        # Restore training mode if it was active
        if is_training:
            self.train()

        return per_layer_kv_cache

    def decode_step(
        self,
        tgt_ids_Bx1xC: torch.Tensor,  # Input tokens for the current step [B, 1, C]
        state: DecoderInferenceState, # Contains caches, positions, etc.
    ) -> torch.Tensor:
        """
        Performs a single decoding step (autoregressive generation).

        Args:
            tgt_ids_Bx1xC: Input token IDs for the current step (shape [B, 1, C]).
            state: The current DecoderInferenceState.

        Returns:
            logits_Bx1xCxV: The output logits for the current step (shape [B, 1, C, V]), cast to float32.
        """
        B, T, C = tgt_ids_Bx1xC.shape
        assert T == 1, "decode_step expects T=1"
        assert C == self.num_channels, "Input channels mismatch"

        # 1. Embed and Combine Channel Tokens
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_Bx1xC[..., i] # Shape [B, 1]
            # Use float32 for embeddings? Or compute_dtype? Let's try compute_dtype.
            channel_embed = self.embeddings[i](channel_tokens).to(self.compute_dtype) # Shape [B, 1, D]
            x = channel_embed if x is None else x + channel_embed
        # x shape: [B, 1, D]
        # Apply dropout if training: x = self.dropout(x)

        # 2. Pass through Decoder Layers
        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(
                x,
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
                prefill=False, # Not prefilling during single step
            ) # Output shape [B, 1, D]

        # 3. Final Normalization
        # Use float32 for norm
        x = self.norm(x.to(torch.float32)).to(self.compute_dtype) # Shape [B, 1, D]

        # 4. Project to Logits
        logits_Bx1xCxV = self.logits_dense(x) # Shape [B, 1, C, V]

        # Return logits in float32 for potentially more stable sampling
        return logits_Bx1xCxV.to(torch.float32)

    def forward(self, tgt_ids_BxTxC: torch.Tensor, state: DecoderInferenceState) -> torch.Tensor:
        """
        Forward pass for the Decoder stack during prefill or training.

        Args:
            tgt_ids_BxTxC: Target token IDs (B, T, C).
            state: DecoderInferenceState containing encoder output, positions, masks, and caches.

        Returns:
            logits: The final output logits (B, T, C, V), cast to float32.
        """
        B, T, C = tgt_ids_BxTxC.shape
        assert C == self.num_channels, "Input channels mismatch"

        # 1. Embed and Combine Channel Tokens
        x = None
        for i in range(self.num_channels):
            channel_tokens = tgt_ids_BxTxC[..., i] # Shape [B, T]
            # Use float32 or compute_dtype? Let's try compute_dtype.
            channel_embed = self.embeddings[i](channel_tokens).to(self.compute_dtype) # Shape [B, T, D]
            x = channel_embed if x is None else x + channel_embed
        # x shape: [B, T, D]
        # Apply dropout if training: x = self.dropout(x)

        # 2. Pass through Decoder Layers (Prefill Mode)
        for i, layer in enumerate(self.layers):
            self_cache = state.self_attn_cache[i]
            cross_cache = state.cross_attn_cache[i]
            x = layer(
                x,
                state,
                self_attn_cache=self_cache,
                cross_attn_cache=cross_cache,
                prefill=True, # Indicate prefill mode
                ) # Output shape [B, T, D]

        # 3. Final Normalization
        # Use float32 for norm
        x = self.norm(x.to(torch.float32)).to(self.compute_dtype) # Shape [B, T, D]

        # 4. Project to Logits
        logits_BxTxCxV = self.logits_dense(x) # Shape [B, T, C, V]

        # Return logits in float32
        return logits_BxTxCxV.to(torch.float32)


class DiaModel(
    nn.Module,
    PyTorchModelHubMixin,
    # config_class=DiaConfig, # Link config class if HF integration needs it
    repo_url="https://github.com/nari-labs/dia",
    pipeline_tag="text-to-speech",
    license="apache-2.0",
    # Define how to serialize/deserialize the custom DiaConfig
    # This might require adjustments based on PyTorchModelHubMixin specifics
    coders={
        DiaConfig: (
            lambda config: config.model_dump(), # Serialize using Pydantic's dump
            lambda data: DiaConfig.model_validate(data), # Deserialize using Pydantic's validate
        ),
    },
):
    """PyTorch Dia Model main module, combining Encoder and Decoder."""

    # Updated __init__ signature
    def __init__(self, config: DiaConfig, compute_dtype: Optional[torch.dtype] = None):
        super().__init__()
        # Store config directly for access via self.config
        # Pydantic models are generally compatible with state_dict if structured well
        self.config = config

        # Determine compute dtype if not provided, infer from config
        if compute_dtype is None:
             try:
                  dtype_str = config.model.weight_dtype
                  if dtype_str == "bfloat16": compute_dtype = torch.bfloat16
                  elif dtype_str == "float16": compute_dtype = torch.float16
                  else: compute_dtype = torch.float32
             except AttributeError: # Fallback if weight_dtype is missing
                  compute_dtype = torch.float32
             print(f"Inferred compute_dtype: {compute_dtype} from config")

        # Initialize submodules, passing the determined compute_dtype
        self.encoder = Encoder(config, compute_dtype)
        self.decoder = Decoder(config, compute_dtype)

    # No forward method defined here as encoder/decoder are called separately in the Dia wrapper
    # If this were intended for direct use like a standard HF model, a forward pass would be needed.

    # Add _keys_to_ignore_on_load_missing potentially if needed for HF compatibility
    # def _keys_to_ignore_on_load_missing(self):
    #     return [...]