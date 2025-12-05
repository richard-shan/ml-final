import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import os
from tqdm import tqdm
import json
from dataclasses import dataclass, asdict
import numpy as np


class SparseAutoencoder(nn.Module):

    def __init__(self, d_in, d_sae):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        
        # Encoder: maps input to sparse features
        # Slightly larger init for normalized data
        self.W_enc = nn.Parameter(torch.randn(d_sae, d_in) * (2.0 / np.sqrt(d_in)))
        
        # Small positive bias to prevent immediate feature death
        self.b_enc = nn.Parameter(torch.zeros(d_sae) + 0.05)
        
        # Decoder: maps sparse features back to input space  
        self.W_dec = nn.Parameter(torch.randn(d_in, d_sae) / np.sqrt(d_sae))
        
        # For pre-normalized data (zero mean), we don't need b_dec for centering
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Initialize decoder to have unit norm columns
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
        
        # Make encoder and decoder roughly transposes (tied init)
        self.W_enc.data = self.W_dec.data.T.clone() * 0.5
        
    def encode(self, x):
        # Encode input to sparse features.
        pre_acts = x @ self.W_enc.T + self.b_enc
        feature_acts = F.relu(pre_acts)
        return feature_acts, pre_acts
    
    def decode(self, feature_acts):
        # Decode sparse features to reconstruction.
        return feature_acts @ self.W_dec.T + self.b_dec
    
    def forward(self, x):
        # Forward pass.
        feature_acts, pre_acts = self.encode(x)
        reconstruction = self.decode(feature_acts)
        
        return {
            'feature_acts': feature_acts,
            'pre_acts': pre_acts,
            'reconstruction': reconstruction
        }
    
    @torch.no_grad()
    def resample_dead_features(self, dead_mask, activations, optimizer):
        # Resample dead features. Method currently bugged due to implementation time 5000 discrete checking. fix if extra time and compute for rerun.
        if not dead_mask.any():
            return 0
        
        n_dead = dead_mask.sum().item()
        dead_indices = torch.where(dead_mask)[0]
        
        # Find high-reconstruction-error examples
        with torch.no_grad():
            feature_acts, _ = self.encode(activations)
            recon = self.decode(feature_acts)
            losses = (activations - recon).pow(2).sum(dim=-1)
            
            # Sample proportionally to loss (not just top-k)
            probs = losses / losses.sum()
            indices = torch.multinomial(probs, min(n_dead, len(activations)), replacement=False)
        
        # Get new directions from high-loss examples
        new_directions = activations[indices[:n_dead]]
        new_directions = F.normalize(new_directions, dim=-1)
        
        # Reinitialize dead features
        # Decoder: point toward high-loss examples
        self.W_dec.data[:, dead_indices[:len(new_directions)]] = new_directions.T
        
        # Encoder: roughly transpose of decoder, scaled down
        self.W_enc.data[dead_indices[:len(new_directions)]] = new_directions * 0.1
        
        # Bias: small positive value
        self.b_enc.data[dead_indices[:len(new_directions)]] = 0.05
        
        # Reset optimizer momentum for resampled features
        # This is important - stale momentum can kill features immediately
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                state = optimizer.state.get(param)
                if state is not None:
                    if 'exp_avg' in state:
                        if param is self.W_enc:
                            state['exp_avg'][dead_indices[:len(new_directions)]] = 0
                            state['exp_avg_sq'][dead_indices[:len(new_directions)]] = 0
                        elif param is self.W_dec:
                            state['exp_avg'][:, dead_indices[:len(new_directions)]] = 0
                            state['exp_avg_sq'][:, dead_indices[:len(new_directions)]] = 0
                        elif param is self.b_enc:
                            state['exp_avg'][dead_indices[:len(new_directions)]] = 0
                            state['exp_avg_sq'][dead_indices[:len(new_directions)]] = 0
        
        return n_dead


@dataclass
class SAEConfig:
    
    # Model/Data
    activation_file: str = "qwen2.5_7b_slimpajama_10M_activations_normalized.pt"
    layer_name: str = "layer_20"
    
    # Architecture
    d_in: int = 3584
    d_sae: int = 3584 * 8 
    
    # Training
    batch_size: int = 4096
    total_steps: int = 20_000  
    lr: float = 1e-4  
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Sparsity - more conservative
    l1_coefficient: float = 0.0008 
    l1_warmup_steps: int = 10000  # extended warmup
    
    # Decoder normalization (soft penalty instead of hard constraint)
    decoder_norm_penalty: float = 0.001
    
    resample_frequency: int = 5000  # Resample every N steps
    dead_feature_threshold: int = 5000  # Features dead for this many steps get resampled
    
    # Learning rate schedule
    lr_warmup_steps: int = 1000  # Added LR warmup
    lr_decay_start: int = 15000  # When to start decay
    lr_decay_steps: int = 5000  # Decay duration
    
    # Logging
    log_frequency: int = 50
    checkpoint_frequency: int = 5000
    wandb_project: str = "qwen2.5-7b-sae"
    wandb_name: str = "qwen2.5-7b-layer20-8x-fixed"
    use_wandb: bool = True
    
    # Paths
    checkpoint_dir: str = "checkpoints_qwen2.5_7b_fixed"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class ActivationDataset(Dataset):
    # Dataset for cached activations with aggressive RAM loading.
    def __init__(self, activation_file, device='cpu', preload_to_gpu=False):
        print(f"Loading activations from {activation_file}...")
        print("Aggressively loading entire file into RAM...")
        
        # Load directly into RAM
        self.activations = torch.load(
            activation_file,
            map_location='cpu',
            weights_only=False
        )
        
        if not isinstance(self.activations, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(self.activations)}")
        
        # Convert to float32 and make contiguous for fastest access
        if self.activations.dtype != torch.float32:
            print(f"  Converting from {self.activations.dtype} to float32...")
            self.activations = self.activations.float()
        
        if not self.activations.is_contiguous():
            print("  Making tensor contiguous...")
            self.activations = self.activations.contiguous()
        
        # Optionally preload to GPU for maximum speed
        self.on_gpu = False
        if preload_to_gpu and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            tensor_size = self.activations.numel() * 4  # float32 = 4 bytes
            
            # Only preload if tensor fits in <50% of GPU memory (leave room for model + gradients)
            if tensor_size < gpu_mem * 0.5:
                print(f"  Preloading to GPU ({tensor_size / 1e9:.2f} GB)...")
                self.activations = self.activations.cuda()
                self.on_gpu = True
            else:
                print(f"  Tensor too large for GPU preload ({tensor_size / 1e9:.2f} GB > {gpu_mem * 0.5 / 1e9:.2f} GB limit)")
        
        print(f"✓ Loaded {len(self):,} activation vectors")
        print(f"  Shape: {self.activations.shape}")
        print(f"  Dtype: {self.activations.dtype}")
        print(f"  Memory: {self.activations.numel() * 4 / 1e9:.2f} GB")
        print(f"  Location: {'GPU' if self.on_gpu else 'RAM'}")
        
        # Verify normalization
        sample = self.activations[:1000]
        print(f"  Sample mean: {sample.mean():.4f} (should be ~0)")
        print(f"  Sample std: {sample.std():.4f} (should be ~1)")
        
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


def get_lr_schedule(step, config):
    # Compute learning rate for current step with warmup and decay
    # Warmup phase
    if step < config.lr_warmup_steps:
        return config.lr * (step / config.lr_warmup_steps)
    
    # Constant phase
    if step < config.lr_decay_start:
        return config.lr
    
    # Decay phase
    if step < config.lr_decay_start + config.lr_decay_steps:
        progress = (step - config.lr_decay_start) / config.lr_decay_steps
        return config.lr * (1 - 0.9 * progress)  # Decay to 10% of max
    
    # Post-decay
    return config.lr * 0.1


def get_l1_coefficient(step, config):
    # Compute L1 coefficient with linear warmup
    if step < config.l1_warmup_steps:
        # Linear warmup (gentler than quadratic)
        progress = step / config.l1_warmup_steps
        return config.l1_coefficient * progress
    return config.l1_coefficient


def compute_metrics(sae_out, x, l1_coeff, sae, config):
    # Compute comprehensive SAE training metrics
    feature_acts = sae_out['feature_acts']
    pre_acts = sae_out['pre_acts']
    reconstruction = sae_out['reconstruction']
    
    # Core losses
    mse_loss = F.mse_loss(reconstruction, x)
    sparsity_loss = l1_coeff * feature_acts.abs().sum(dim=-1).mean()
    
    # Soft decoder normalization penalty (instead of hard constraint)
    decoder_norms = torch.norm(sae.W_dec, dim=0)
    decoder_norm_loss = config.decoder_norm_penalty * (decoder_norms - 1).pow(2).mean()
    
    total_loss = mse_loss + sparsity_loss + decoder_norm_loss
    
    # Sparsity metrics
    l0 = (feature_acts > 0).float().sum(dim=-1).mean()
    l1_norm = feature_acts.abs().sum(dim=-1).mean()
    
    # Reconstruction quality
    x_var = x.var()
    explained_var = 1 - (mse_loss / (x_var + 1e-8))
    
    with torch.no_grad():
        # Pre-activation stats (diagnostic)
        pre_act_mean = pre_acts.mean()
        pre_act_std = pre_acts.std()
        pre_act_positive_frac = (pre_acts > 0).float().mean()
        
        # Variance metrics
        total_variance = torch.var(x, dim=0).sum()
        residual_variance = torch.var(x - reconstruction, dim=0).sum()
        frac_variance_explained = 1 - (residual_variance / (total_variance + 1e-8))
        mse_loss_normalized = mse_loss / (x_var + 1e-8)
        
        # Feature sparsity tracking
        feature_sparsity = (feature_acts > 0).float().mean(dim=0)
        dead_features = (feature_sparsity == 0).sum()
        frac_dead = dead_features.float() / feature_acts.shape[1]
        
        # Mean log10 feature sparsity
        feature_sparsity_safe = torch.clamp(feature_sparsity, min=1e-10)
        mean_log10_feature_sparsity = torch.log10(feature_sparsity_safe).mean()
        
        # Feature activation stats
        mean_feature_act = feature_acts.mean()
        max_feature_act = feature_acts.max()
        
        # L2 norms
        l2_norm_in = torch.norm(x, dim=-1).mean()
        l2_norm_out = torch.norm(reconstruction, dim=-1).mean()
        l2_ratio = l2_norm_out / (l2_norm_in + 1e-8)
        relative_reconstruction_bias = (l2_norm_out - l2_norm_in) / (l2_norm_in + 1e-8)
        
        # Weight norms
        W_enc_norm = torch.norm(sae.W_enc, dim=1).mean()
        W_dec_norm_mean = decoder_norms.mean()
        W_dec_norm_std = decoder_norms.std()
        b_enc_mean = sae.b_enc.mean()
        b_enc_norm = torch.norm(sae.b_enc)
        
    return {
        # Primary losses
        'loss': total_loss,
        'mse_loss': mse_loss,
        'sparsity_loss': sparsity_loss,
        'decoder_norm_loss': decoder_norm_loss,
        
        # Sparsity metrics
        'l0': l0,
        'l1_norm': l1_norm,
        'mean_log10_feature_sparsity': mean_log10_feature_sparsity,
        
        # Reconstruction quality
        'explained_variance': explained_var,
        'frac_variance_explained': frac_variance_explained,
        'mse_loss_normalized': mse_loss_normalized,
        
        # Shrinkage metrics
        'l2_ratio': l2_ratio,
        'relative_reconstruction_bias': relative_reconstruction_bias,
        
        # Feature health
        'dead_features': dead_features,
        'frac_dead': frac_dead,
        'feature_sparsity': feature_sparsity,
        
        # Feature activation stats
        'mean_feature_act': mean_feature_act,
        'max_feature_act': max_feature_act,
        
        # Pre-activation diagnostics
        'pre_act_mean': pre_act_mean,
        'pre_act_std': pre_act_std,
        'pre_act_positive_frac': pre_act_positive_frac,
        
        # Weights
        'W_enc_norm': W_enc_norm,
        'W_dec_norm_mean': W_dec_norm_mean,
        'W_dec_norm_std': W_dec_norm_std,
        'W_dec_norms': decoder_norms,
        'b_enc_mean': b_enc_mean,
        'b_enc_norm': b_enc_norm,
        
        # Scale tracking
        'l2_norm_in': l2_norm_in,
        'l2_norm_out': l2_norm_out,
    }


def train_sae(config):
    # main training loop    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(config.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # initialize W&B for tracking (force relogin)
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=asdict(config)
        )
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data")
    print("="*60)
    dataset = ActivationDataset(config.activation_file, preload_to_gpu=True)
    # If data is already on GPU, use simpler DataLoader settings
    if dataset.on_gpu:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # No workers needed for GPU data
            pin_memory=False,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=8,  # More workers for faster loading
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,  # Prefetch more batches
        )
    
    # Create model
    print("\n" + "="*60)
    print("Creating SAE")
    print("="*60)
    print(f"Model: Qwen/Qwen2.5-7B")
    print(f"Input dimension: {config.d_in:,}")
    print(f"SAE dimension: {config.d_sae:,}")
    print(f"Expansion factor: {config.d_sae / config.d_in:.1f}x")
    print(f"Total parameters: {(config.d_in * config.d_sae * 2 + config.d_in + config.d_sae):,}")
    
    sae = SparseAutoencoder(config.d_in, config.d_sae).to(config.device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Training")
    print("="*60)
    print(f"Total steps: {config.total_steps:,}")
    print(f"Batch size: {config.batch_size:,}")
    print(f"Total tokens: {config.total_steps * config.batch_size:,}")
    print(f"L1 coefficient: {config.l1_coefficient}")
    print(f"L1 warmup steps: {config.l1_warmup_steps}")
    print(f"Resample frequency: {config.resample_frequency}")
    print("="*60 + "\n")
    
    data_iter = iter(dataloader)
    
    # Track when features last fired
    last_fired_step = torch.zeros(config.d_sae, dtype=torch.long, device=config.device)
    total_resampled = 0
    
    # Keep a buffer for resampling
    resample_buffer = None
    
    for step in tqdm(range(config.total_steps), desc="Training"):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        batch = batch.to(config.device)
        
        # Store batch for potential resampling
        if step % config.resample_frequency == 0:
            resample_buffer = batch.clone()
        
        # Update learning rate
        lr = get_lr_schedule(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get current L1 coefficient
        l1_coeff = get_l1_coefficient(step, config)
        
        # Forward pass
        sae_out = sae(batch)
        
        # Update last_fired tracking
        with torch.no_grad():
            feature_fired = (sae_out['feature_acts'] > 0).any(dim=0)
            last_fired_step[feature_fired] = step
        
        # Compute metrics and loss
        metrics = compute_metrics(sae_out, batch, l1_coeff, sae, config)
        
        # Backward pass
        optimizer.zero_grad()
        metrics['loss'].backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        
        optimizer.step()
        
        # bugged dead feature resampling
        n_resampled = 0
        if (step + 1) % config.resample_frequency == 0 and step > config.l1_warmup_steps:
            steps_since_fired = step - last_fired_step
            dead_mask = steps_since_fired > config.dead_feature_threshold
            
            if dead_mask.any() and resample_buffer is not None:
                n_resampled = sae.resample_dead_features(dead_mask, resample_buffer, optimizer)
                total_resampled += n_resampled
                
                # Reset last_fired for resampled features
                last_fired_step[dead_mask] = step
                
                print(f"\nResampled {n_resampled:,} dead features (total: {total_resampled:,})")
        
        # Logging
        if step % config.log_frequency == 0:
            steps_since_fired = step - last_fired_step
            mean_steps_since_fired = steps_since_fired.float().mean()
            
            log_metrics = {
                # Details
                'details/n_training_samples': step * config.batch_size,
                'details/l1_coefficient': l1_coeff,
                'details/current_learning_rate': lr,
                
                # Primary losses
                'loss/total': metrics['loss'].item(),
                'loss/mse': metrics['mse_loss'].item(),
                'loss/sparsity': metrics['sparsity_loss'].item(),
                'loss/decoder_norm': metrics['decoder_norm_loss'].item(),
                
                # Loss ratio (diagnostic)
                'loss/mse_to_sparsity_ratio': metrics['mse_loss'].item() / (metrics['sparsity_loss'].item() + 1e-8),
                
                # Metrics
                'metrics/mean_log10_feature_sparsity': metrics['mean_log10_feature_sparsity'].item(),
                'metrics/l0': metrics['l0'].item(),
                'metrics/explained_variance': metrics['explained_variance'].item(),
                
                # Sparsity
                'sparsity/l0': metrics['l0'].item(),
                'sparsity/l1': metrics['l1_norm'].item(),
                'sparsity/frac_dead': metrics['frac_dead'].item(),
                'sparsity/dead_features': metrics['dead_features'].item(),
                'sparsity/mean_steps_since_fired': mean_steps_since_fired.item(),
                'sparsity/total_resampled': total_resampled,
                
                # Shrinkage
                'shrinkage/relative_reconstruction_bias': metrics['relative_reconstruction_bias'].item(),
                'shrinkage/l2_ratio': metrics['l2_ratio'].item(),
                
                # Reconstruction quality
                'reconstruction/explained_variance': metrics['explained_variance'].item(),
                'reconstruction/frac_variance_explained': metrics['frac_variance_explained'].item(),
                'reconstruction/mse_normalized': metrics['mse_loss_normalized'].item(),
                
                # Pre-activation diagnostics
                'diagnostics/pre_act_mean': metrics['pre_act_mean'].item(),
                'diagnostics/pre_act_std': metrics['pre_act_std'].item(),
                'diagnostics/pre_act_positive_frac': metrics['pre_act_positive_frac'].item(),
                
                # Feature activation stats
                'features/mean_act': metrics['mean_feature_act'].item(),
                'features/max_act': metrics['max_feature_act'].item(),
                
                # Weights
                'weights/W_enc_norm': metrics['W_enc_norm'].item(),
                'weights/W_dec_norm_mean': metrics['W_dec_norm_mean'].item(),
                'weights/W_dec_norm_std': metrics['W_dec_norm_std'].item(),
                'weights/b_enc_mean': metrics['b_enc_mean'].item(),
                'weights/b_enc_norm': metrics['b_enc_norm'].item(),
                
                # Scale tracking
                'scale/l2_norm_in': metrics['l2_norm_in'].item(),
                'scale/l2_norm_out': metrics['l2_norm_out'].item(),
            }
            
            if config.use_wandb:
                wandb.log(log_metrics, step=step)
                
                # Log histograms less frequently
                if step % (config.log_frequency * 20) == 0:
                    wandb.log({
                        "weights/W_dec_norms_hist": wandb.Histogram(
                            metrics['W_dec_norms'].detach().cpu().numpy()
                        ),
                    }, step=step)
            
            # Print detailed status periodically
            if step % (config.log_frequency * 10) == 0:
                print(f"\n{'='*70}")
                print(f"Step {step:5d}")
                print(f"{'='*70}")
                print(f"  Loss: {metrics['loss'].item():.4f} "
                      f"(MSE: {metrics['mse_loss'].item():.4f}, "
                      f"L1: {metrics['sparsity_loss'].item():.4f}, "
                      f"Dec: {metrics['decoder_norm_loss'].item():.4f})")
                print(f"  MSE/Sparsity ratio: {metrics['mse_loss'].item() / (metrics['sparsity_loss'].item() + 1e-8):.2f}")
                print(f"  L0: {metrics['l0'].item():.1f} | "
                      f"ExpVar: {metrics['explained_variance'].item():.3f} | "
                      f"Dead: {metrics['frac_dead'].item():.3f}")
                print(f"  Pre-act: mean={metrics['pre_act_mean'].item():.3f}, "
                      f"std={metrics['pre_act_std'].item():.3f}, "
                      f"pos_frac={metrics['pre_act_positive_frac'].item():.3f}")
                print(f"  L1 coeff: {l1_coeff:.5f} | LR: {lr:.2e}")
        
        # Checkpointing
        if (step + 1) % config.checkpoint_frequency == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, 
                f'sae_step_{step+1}.pt'
            )
            torch.save({
                'step': step,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'last_fired_step': last_fired_step.cpu(),
                'total_resampled': total_resampled,
                'config': asdict(config),
            }, checkpoint_path)
            print(f"\n✓ Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config.checkpoint_dir, 'sae_final.pt')
    torch.save({
        'step': config.total_steps,
        'model_state_dict': sae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'last_fired_step': last_fired_step.cpu(),
        'total_resampled': total_resampled,
        'config': asdict(config),
    }, final_path)
    print(f"\n✓ Saved final model to {final_path}")
    
    if config.use_wandb:
        wandb.finish()
    
    return sae


if __name__ == "__main__":
    config = SAEConfig()
    
    # Check if activation file exists
    if not os.path.exists(config.activation_file):
        print(f"\nError: Activation file not found: {config.activation_file}")
        exit(1)
    
    # train sae
    sae = train_sae(config)
    
    print("Training Complete!")
    print(f"Final model saved to: {config.checkpoint_dir}/sae_final.pt")