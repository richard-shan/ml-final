import torch
import numpy as np
import os

# Configuration
INPUT_FILE = "qwen2.5_7b_slimpajama_10M_activations.pt"
OUTPUT_FILE = "qwen2.5_7b_slimpajama_10M_activations_normalized.pt"
REMOVE_OUTLIERS = True
OUTLIER_THRESHOLD = 5.0  # Remove vectors with norm > threshold * median

print("="*60)
print("Activation Normalization (Aggressive Loading)")
print("="*60)
print("Loading entire file into memory for fast processing...")

# aggressive full loading since I have enough RAM
print(f"\nLoading: {INPUT_FILE}")
acts = torch.load(INPUT_FILE, map_location='cpu')

print(f"Loaded into RAM")
print(f"Original shape: {acts.shape}")
print(f"Original dtype: {acts.dtype}")
print(f"Memory usage: ~{acts.element_size() * acts.nelement() / (1024**3):.1f} GB")

# Statistics before
print("\n" + "="*60)
print("BEFORE Normalization:")
print("="*60)
print(f"Mean: {acts.mean():.6f}")
print(f"Std: {acts.std():.6f}")
print(f"Min: {acts.min():.6f}")
print(f"Max: {acts.max():.6f}")

# Compute norms
print("\nComputing norms...")
norms = torch.norm(acts, dim=1)
print(f"Computed {len(norms):,} norms")

print(f"\nNorm statistics:")
print(f"  Mean norm: {norms.mean():.6f}")
print(f"  Median norm: {norms.median():.6f}")
print(f"  Std norm: {norms.std():.6f}")
print(f"  Min norm: {norms.min():.6f}")
print(f"  Max norm: {norms.max():.6f}")

# Find and remove outliers
if REMOVE_OUTLIERS:
    print("\n" + "="*60)
    print("Outlier Detection & Removal:")
    print("="*60)
    
    median_norm = norms.median()
    threshold = median_norm * OUTLIER_THRESHOLD
    outlier_mask = norms > threshold
    n_outliers = outlier_mask.sum().item()
    pct_outliers = 100 * n_outliers / len(acts)
    
    print(f"Median norm: {median_norm:.2f}")
    print(f"Threshold (>{OUTLIER_THRESHOLD}x median): {threshold:.2f}")
    print(f"Outliers found: {n_outliers:,} ({pct_outliers:.2f}%)")
    
    if n_outliers > 0:
        print(f"Outlier norms: min={norms[outlier_mask].min():.2f}, max={norms[outlier_mask].max():.2f}")
        
        # Remove outliers 
        print(f"\nRemoving outliers...")
        acts = acts[~outlier_mask]
        print(f"Removed {n_outliers:,} outliers")
        print(f"New shape: {acts.shape}")
        print(f"New memory usage: ~{acts.element_size() * acts.nelement() / (1024**3):.1f} GB")
    else:
        n_outliers = 0
else:
    n_outliers = 0


# # Standardize (zero mean, unit variance per dimension)
# Compute statistics in one pass
mean = acts.mean(dim=0, keepdim=True)
std = acts.std(dim=0, keepdim=True)

print(f"  Mean shape: {mean.shape}")
print(f"  Std shape: {std.shape}")

# Avoid division by zero
std_clamped = torch.clamp(std, min=1e-6)
n_zero_std = (std < 1e-6).sum().item()
if n_zero_std > 0:
    print(f"  Warning: {n_zero_std} dimensions had near-zero std (clamped to 1e-6)")

print("\nApplying normalization: (x - mean) / std")
acts_normalized = (acts - mean) / std_clamped
print(f"Normalized all {len(acts_normalized):,} vectors")

# Statistics after
print("\n" + "="*60)
print("AFTER Normalization:")
print("="*60)
print(f"Mean: {acts_normalized.mean():.6f}")
print(f"Std: {acts_normalized.std():.6f}")
print(f"Min: {acts_normalized.min():.6f}")
print(f"Max: {acts_normalized.max():.6f}")

print("\nRecomputing norms...")
norms_normalized = torch.norm(acts_normalized, dim=1)
print(f"Computed normalized norms")

print(f"\nNorm statistics:")
print(f"  Mean norm: {norms_normalized.mean():.6f}")
print(f"  Median norm: {norms_normalized.median():.6f}")
print(f"  Std norm: {norms_normalized.std():.6f}")
print(f"  Min norm: {norms_normalized.min():.6f}")
print(f"  Max norm: {norms_normalized.max():.6f}")

# Sample checks
print("\n" + "="*60)
print("Sample vector norms (first 10):")
print("="*60)
for i in range(min(10, len(acts_normalized))):
    new_norm = torch.norm(acts_normalized[i])
    print(f"  Vector {i}: {new_norm:>6.2f}")

# Additional validation
print("\n" + "="*60)
print("Validation:")
print("="*60)

# Check for NaN or Inf
has_nan = torch.isnan(acts_normalized).any()
has_inf = torch.isinf(acts_normalized).any()
print(f"Contains NaN: {has_nan}")
print(f"Contains Inf: {has_inf}")

if has_nan or has_inf:
    print("WARNING: Found NaN or Inf values!")
else:
    print("No NaN or Inf values")

# Save
print("\n" + "="*60)
print("Saving:")
print("="*60)
print(f"Saving to: {OUTPUT_FILE}")

# Save normalized activations
torch.save(acts_normalized, OUTPUT_FILE)
print(f"Saved normalized activations")

# Save normalization parameters (for denormalizing later if needed)
print(f"\nSaving normalization parameters...")
normalization_params = {
    'mean': mean,
    'std': std,
    'std_clamped': std_clamped,
    'removed_outliers': REMOVE_OUTLIERS,
    'outlier_threshold': OUTLIER_THRESHOLD,
    'n_outliers_removed': n_outliers,
    'original_shape': torch.Size([len(acts) + n_outliers, acts.shape[1]]),
    'final_shape': acts_normalized.shape,
}

param_file = OUTPUT_FILE.replace('.pt', '_params.pt')
torch.save(normalization_params, param_file)
print(f"Saved normalization params to: {param_file}")

# File size comparison
orig_size = os.path.getsize(INPUT_FILE) / (1024**3)
new_size = os.path.getsize(OUTPUT_FILE) / (1024**3)

print("\n" + "="*60)
print("Summary:")
print("="*60)
print(f"Original vectors: {10_000_000:,}")
print(f"Final vectors: {len(acts_normalized):,}")
if REMOVE_OUTLIERS and n_outliers > 0:
    print(f"Removed: {n_outliers:,} outliers ({100*n_outliers/(len(acts_normalized)+n_outliers):.2f}%)")
print(f"\nFile sizes:")
print(f"  Original: {orig_size:.1f} GB")
print(f"  Normalized: {new_size:.1f} GB")
