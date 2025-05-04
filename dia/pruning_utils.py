import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import List, Tuple, Type

# Import the custom layer used in DiaModel
from .layers import DenseGeneral

# --- Updated Default Module Types ---
# Now includes DenseGeneral by default
DEFAULT_PRUNABLE_MODULES = (DenseGeneral, nn.Linear, nn.Conv1d)

def get_prunable_modules(model: nn.Module, module_types: Tuple[Type[nn.Module], ...] = DEFAULT_PRUNABLE_MODULES) -> List[Tuple[nn.Module, str]]:
    """
    Identifies modules of specified types within the model that have a 'weight' parameter.

    Args:
        model: The model to inspect.
        module_types: A tuple of module classes to consider for pruning.
                      Defaults to (DenseGeneral, nn.Linear, nn.Conv1d).

    Returns:
        A list of tuples, where each tuple contains (module, 'weight').
    """
    params_to_prune = []
    print(f"Searching for prunable modules of types: {[m.__name__ for m in module_types]}")
    found_count = 0
    for name, module in model.named_modules(): # Use named_modules for better debugging
        if isinstance(module, module_types):
            if hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter):
                params_to_prune.append((module, 'weight'))
                found_count += 1
                # print(f"  Found prunable module: {name} ({module.__class__.__name__})")
            # else:
                # print(f"  Skipping module (no weight parameter): {name} ({module.__class__.__name__})")
        # else:
            # print(f"  Skipping module (wrong type): {name} ({module.__class__.__name__})")

    print(f"Found {found_count} modules with 'weight' parameters matching the specified types.")
    return params_to_prune

def apply_unstructured_pruning(model: nn.Module, amount: float, module_types: Tuple[Type[nn.Module], ...] = DEFAULT_PRUNABLE_MODULES):
    """
    Applies global unstructured L1 magnitude pruning to specified module types.

    Args:
        model: The model to prune.
        amount: The fraction of weights to prune (0.0 to 1.0).
        module_types: Module types to prune. Defaults includes DenseGeneral.
    """
    parameters_to_prune = get_prunable_modules(model, module_types)
    if not parameters_to_prune:
        print("Warning: No prunable modules found for unstructured pruning.")
        return

    print(f"Applying global unstructured L1 pruning with amount: {amount}")
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    print("Unstructured pruning applied.")

def apply_structured_pruning(model: nn.Module, amount: float, dim: int, n: int = 2, module_types: Tuple[Type[nn.Module], ...] = DEFAULT_PRUNABLE_MODULES):
    """
    Applies global structured Ln magnitude pruning along a specified dimension.
    Commonly used for pruning filters (dim=0 in Conv) or neurons (dim=0 in Linear output).

    Args:
        model: The model to prune.
        amount: The fraction of structures (e.g., channels, neurons) to prune (0.0 to 1.0).
        dim: The dimension along which to prune (e.g., 0 for output features in DenseGeneral).
        n: The norm order (e.g., 1 for L1, 2 for L2).
        module_types: Module types to prune. Defaults includes DenseGeneral.
    """
    parameters_to_prune = get_prunable_modules(model, module_types)
    if not parameters_to_prune:
        print("Warning: No prunable modules found for structured pruning.")
        return

    print(f"Applying structured L{n} pruning with amount: {amount} along dim: {dim} (Layer-wise)")
    # Note: PyTorch's global_structured applies the *same mask* globally based on overall scores.
    # For layer-wise structured pruning (more common), iterate and apply prune.ln_structured individually.
    num_pruned_total = 0
    num_params_total = 0
    pruned_module_count = 0
    skipped_module_count = 0

    for module, name in parameters_to_prune:
        # Check if the specified dimension exists for the module's weight
        if dim >= module.weight.dim():
            print(f" - Skipping {module.__class__.__name__}: Pruning dim {dim} >= weight dim {module.weight.dim()}")
            skipped_module_count += 1
            continue

        try:
            prune.ln_structured(module, name=name, amount=amount, n=n, dim=dim)
            # Verify pruning actually happened (check for mask)
            if prune.is_pruned(module):
                 num_pruned = torch.sum(module.weight == 0).item() # Check actual zeros after applying mask
                 num_params = module.weight.nelement()
                 num_pruned_total += num_pruned
                 num_params_total += num_params
                 pruned_module_count += 1
                 # print(f" - Pruned {module.__class__.__name__}: {num_pruned}/{num_params} params zeroed.")
            else:
                 print(f" - Warning: Applied structured pruning to {module.__class__.__name__}, but prune.is_pruned is False.")
                 skipped_module_count += 1

        except Exception as e:
            print(f" - Error pruning {module.__class__.__name__} along dim {dim}: {e}")
            skipped_module_count += 1

    final_sparsity = num_pruned_total / num_params_total if num_params_total > 0 else 0
    print(f"Structured pruning applied to {pruned_module_count} modules, skipped {skipped_module_count}.")
    if num_params_total > 0:
        print(f"Overall sparsity achieved in targeted layers: {final_sparsity:.4f} ({num_pruned_total}/{num_params_total} zero parameters)")
    else:
        print("No parameters were pruned.")


def make_pruning_permanent(model: nn.Module, module_types: Tuple[Type[nn.Module], ...] = DEFAULT_PRUNABLE_MODULES):
    """
    Removes the pruning re-parameterization from the model, making the pruned weights permanent zeros.

    Args:
        model: The model to modify.
        module_types: Module types where pruning might have been applied. Defaults includes DenseGeneral.
    """
    print("Making pruning permanent...")
    count = 0
    for module in model.modules():
        # Check if the module is one of the types we might have pruned
        if isinstance(module, module_types):
            # Check if the 'weight' parameter specifically was pruned
            if prune.is_pruned(module): # General check first
                 # Check if 'weight_orig' exists, indicating pruning on 'weight'
                 # This check might be fragile depending on pruning implementation details
                 is_weight_pruned = False
                 for hook in module._forward_pre_hooks.values():
                      if isinstance(hook, prune.BasePruningMethod) and hook._tensor_name == "weight":
                           is_weight_pruned = True
                           break
                 if is_weight_pruned:
                      prune.remove(module, 'weight')
                      count += 1
                      # print(f"  Removed pruning from 'weight' in {module.__class__.__name__}")
                 # else:
                      # print(f"  Module {module.__class__.__name__} is pruned, but not on 'weight'. Skipping remove.")

    print(f"Pruning made permanent for 'weight' parameter in {count} modules.")

def check_pruning_sparsity(model: nn.Module, module_types: Tuple[Type[nn.Module], ...] = DEFAULT_PRUNABLE_MODULES) -> float:
    """
    Calculates the global sparsity of the specified module types in the model.

    Args:
        model: The model to check.
        module_types: Module types to include in the sparsity calculation. Defaults includes DenseGeneral.

    Returns:
        The global sparsity fraction (0.0 to 1.0).
    """
    total_params = 0
    zero_params = 0
    checked_modules = 0
    for module in model.modules():
        if isinstance(module, module_types):
             # Check if weight exists and is a parameter (might be buffer if permanent)
            weight_param = getattr(module, 'weight', None)
            if weight_param is not None and isinstance(weight_param, (nn.Parameter, torch.Tensor)):
                total_params += weight_param.nelement()
                zero_params += torch.sum(weight_param == 0).item()
                checked_modules += 1

    sparsity = zero_params / total_params if total_params > 0 else 0.0
    print(f"Checked {checked_modules} modules of types {[m.__name__ for m in module_types]}.")
    print(f"Global sparsity in specified layers: {sparsity:.4f} ({zero_params}/{total_params} zero parameters)")
    return sparsity