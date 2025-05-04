# dia/pruning_utils.py

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Type

def get_prunable_modules(model: nn.Module, module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv1d)) -> List[Tuple[nn.Module, str]]:
    """
    Identifies modules of specified types within the model that have a 'weight' parameter.

    Args:
        model: The model to inspect.
        module_types: A tuple of module classes to consider for pruning (e.g., (nn.Linear, nn.Conv1d)).

    Returns:
        A list of tuples, where each tuple contains (module, 'weight').
    """
    params_to_prune = []
    for module in model.modules():
        if isinstance(module, module_types) and hasattr(module, 'weight'):
            params_to_prune.append((module, 'weight'))
    return params_to_prune

def apply_unstructured_pruning(model: nn.Module, amount: float, module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv1d)):
    """
    Applies global unstructured L1 magnitude pruning to specified module types.

    Args:
        model: The model to prune.
        amount: The fraction of weights to prune (0.0 to 1.0).
        module_types: Module types to prune.
    """
    parameters_to_prune = get_prunable_modules(model, module_types)
    if not parameters_to_prune:
        print("Warning: No prunable modules found.")
        return

    print(f"Applying global unstructured L1 pruning with amount: {amount}")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    print("Unstructured pruning applied.")

def apply_structured_pruning(model: nn.Module, amount: float, dim: int, n: int = 2, module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv1d)):
    """
    Applies global structured Ln magnitude pruning along a specified dimension.
    Commonly used for pruning filters (dim=0 in Conv) or neurons (dim=0 in Linear output).

    Args:
        model: The model to prune.
        amount: The fraction of structures (e.g., channels, neurons) to prune (0.0 to 1.0).
        dim: The dimension along which to prune (e.g., 0 for output channels/neurons, 1 for input channels).
        n: The norm order (e.g., 1 for L1, 2 for L2).
        module_types: Module types to prune.
    """
    parameters_to_prune = get_prunable_modules(model, module_types)
    if not parameters_to_prune:
        print("Warning: No prunable modules found.")
        return

    print(f"Applying global structured L{n} pruning with amount: {amount} along dim: {dim}")
    # Note: PyTorch's global_structured applies the *same mask* globally based on overall scores.
    # For layer-wise structured pruning, iterate and apply prune.ln_structured individually.
    # We'll stick to layer-wise for more typical structured pruning:
    num_pruned_total = 0
    num_params_total = 0
    for module, name in parameters_to_prune:
        try:
            prune.ln_structured(module, name=name, amount=amount, n=n, dim=dim)
            num_pruned = torch.sum(module.weight == 0).item()
            num_params = module.weight.nelement()
            num_pruned_total += num_pruned
            num_params_total += num_params
            # print(f" - Pruned {module.__class__.__name__}: {num_pruned}/{num_params} params zeroed.")
        except Exception as e:
            print(f" - Could not prune {module.__class__.__name__} along dim {dim}: {e}")

    final_sparsity = num_pruned_total / num_params_total if num_params_total > 0 else 0
    print(f"Structured pruning applied. Overall sparsity achieved in targeted layers: {final_sparsity:.4f}")


def make_pruning_permanent(model: nn.Module, module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv1d)):
    """
    Removes the pruning re-parameterization from the model, making the pruned weights permanent zeros.

    Args:
        model: The model to modify.
        module_types: Module types where pruning might have been applied.
    """
    print("Making pruning permanent...")
    count = 0
    for module in model.modules():
        if isinstance(module, module_types):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
                count += 1
    print(f"Pruning made permanent for {count} modules.")

def check_pruning_sparsity(model: nn.Module, module_types: Tuple[Type[nn.Module], ...] = (nn.Linear, nn.Conv1d)) -> float:
    """
    Calculates the global sparsity of the specified module types in the model.

    Args:
        model: The model to check.
        module_types: Module types to include in the sparsity calculation.

    Returns:
        The global sparsity fraction (0.0 to 1.0).
    """
    total_params = 0
    zero_params = 0
    for module in model.modules():
        if isinstance(module, module_types) and hasattr(module, 'weight'):
            total_params += module.weight.nelement()
            zero_params += torch.sum(module.weight == 0).item()

    sparsity = zero_params / total_params if total_params > 0 else 0.0
    print(f"Global sparsity in specified layers ({[m.__name__ for m in module_types]}): {sparsity:.4f} ({zero_params}/{total_params} zero parameters)")
    return sparsity
```