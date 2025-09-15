# ConvDecoder Hyperparameter Analysis

## Overview
This document provides a comprehensive analysis of the ConvDecoder hyperparameters, ranked by importance for different use cases, and explains the architectural improvements made to the decoder.

## Hyperparameter Ranking (Most to Least Important)

### 1. **Critical Parameters (Highest Impact)**

#### `latent_dim` (Most Important)
- **Impact**: Determines the bottleneck capacity and representation power
- **Typical Values**: 32-512 for images, 64-128 for Fashion-MNIST
- **Guidance**: 
  - Too small: Underfitting, poor reconstruction
  - Too large: Overfitting, computational waste
  - Rule of thumb: Start with √(H×W×C) where H,W,C are image dimensions

#### `hidden_dims` (Critical)
- **Impact**: Controls architectural capacity and feature hierarchy
- **Typical Values**: `[512, 256, 128, 64]` for 28×28 images
- **Guidance**:
  - Should decrease from latent to output
  - Each layer typically halves the channels
  - Deeper networks: better features but more parameters

#### `activation` (Critical)
- **Impact**: Determines non-linearity and gradient flow
- **Recommended**: 
  - `'relu'`: Fast, stable, good default
  - `'gelu'`: Better for larger models
  - `'swish'`: Good for deeper networks
  - `'mish'`: Often superior but slower
- **Avoid**: `'sigmoid'`, `'tanh'` in hidden layers (vanishing gradients)

### 2. **High Impact Parameters**

#### `final_activation` (High Impact)
- **Impact**: Determines output range and training dynamics
- **Guidance**:
  - `'sigmoid'`: For normalized [0,1] images
  - `'tanh'`: For [-1,1] normalized images  
  - `'identity'`: For raw pixel values with proper loss
- **Critical**: Must match your data preprocessing

#### `normalization` (High Impact)
- **Impact**: Training stability and convergence speed
- **Options**: `'batch'`, `'layer'`, `'group'`, `'instance'`, `'none'`
- **Recommendations**:
  - `'batch'`: Best for large batches (>16)
  - `'group'`: Good compromise, stable across batch sizes
  - `'layer'`: Better for small batches
  - `'instance'`: Good for style transfer

#### `output_size` (High Impact)
- **Impact**: Output resolution, must match target images
- **Typical**: `(28, 28)` for Fashion-MNIST, `(32, 32)` for CIFAR-10
- **Note**: Architecture automatically adjusts for different sizes

### 3. **Medium Impact Parameters**

#### `kernel_size` (Medium-High Impact)
- **Impact**: Receptive field and spatial relationships
- **Default**: 3 (good balance of context and efficiency)
- **Options**: 
  - 3: Standard, efficient
  - 5: More context, slower
  - 1: Pointwise convolutions

#### `upsampling_mode` (Medium Impact)
- **Impact**: Upsampling quality and artifacts
- **Options**:
  - `'transpose'`: Learnable, can create checkerboard artifacts
  - `'bilinear'`: Smooth, no artifacts
  - `'nearest'`: Sharp edges, pixelated
  - `'bicubic'`: Smoothest, slowest
- **Recommendation**: `'bilinear'` for most cases

#### `stride` (Medium Impact)
- **Impact**: Upsampling factor per layer
- **Default**: 2 (doubles resolution)
- **Note**: Only used with transpose convolutions

#### `dropout_rate` (Medium Impact)
- **Impact**: Regularization and overfitting prevention
- **Range**: 0.0-0.5
- **Guidance**:
  - 0.0: No regularization
  - 0.1-0.2: Light regularization
  - 0.3-0.5: Heavy regularization (usually too much for decoders)

### 4. **Lower Impact Parameters**

#### `use_spectral_norm` (Medium-Low Impact)
- **Impact**: Training stability for adversarial settings
- **Use Case**: GANs, when training instability occurs
- **Default**: False (adds computational overhead)

#### `use_self_attention` (Low-Medium Impact)
- **Impact**: Long-range dependencies, mainly for larger images
- **Use Case**: High resolution (>64×64), complex spatial relationships
- **Default**: False (computationally expensive)

#### `attention_resolution` (Low Impact)
- **Impact**: When to apply attention (only if `use_self_attention=True`)
- **Default**: 16 (apply attention at 16×16 and higher)

### 5. **Fine-tuning Parameters (Lowest Impact)**

#### `padding`, `output_padding`, `bias`
- **Impact**: Minor architectural details
- **Defaults**: Usually optimal
- **Only modify**: For specific architectural experiments

#### `init_size`
- **Impact**: Starting spatial resolution
- **Default**: Auto-calculated based on output_size
- **Manual setting**: Rarely needed

## Architectural Features

### New Activation Functions
The decoder now supports 20+ PyTorch activation functions:

**Primary Recommendations:**
- `'relu'`: Fast, stable, good baseline
- `'gelu'`: Better for transformer-style architectures
- `'swish'`/`'silu'`: Good for deeper networks
- `'mish'`: Often superior performance
- `'leaky_relu'`: Prevents dead neurons

**Advanced Options:**
- `'prelu'`: Learnable negative slope
- `'elu'`, `'selu'`: Self-normalizing properties
- `'hardswish'`: Mobile-optimized version of Swish
- `'relu6'`: Bounded ReLU for quantization

### Normalization Options
- **Batch Normalization**: Standard choice for most cases
- **Group Normalization**: Better for varying batch sizes
- **Layer Normalization**: Good for small batches
- **Instance Normalization**: Useful for style transfer
- **No Normalization**: When using spectral normalization

### Self-Attention Module
- Adds long-range spatial dependencies
- Computationally expensive (O(HW)² complexity)
- Most beneficial for high-resolution images (>64×64)
- Applied at specified resolution thresholds

### Spectral Normalization
- Constrains weight matrices to have spectral norm ≤ 1
- Improves training stability in adversarial settings
- Adds computational overhead
- Most useful for GANs and unstable training

## Recommended Configurations

### Fashion-MNIST (28×28)
```python
ConvDecoder(
    latent_dim=64,
    hidden_dims=[256, 128, 64],
    activation='relu',
    final_activation='sigmoid',
    normalization='batch',
    upsampling_mode='bilinear',
    dropout_rate=0.1
)
```

### CIFAR-10 (32×32)
```python
ConvDecoder(
    latent_dim=128,
    hidden_dims=[512, 256, 128, 64],
    activation='gelu',
    final_activation='sigmoid',
    normalization='group',
    upsampling_mode='bilinear',
    dropout_rate=0.1
)
```

### High-Resolution Images (128×128+)
```python
ConvDecoder(
    latent_dim=256,
    hidden_dims=[1024, 512, 256, 128, 64],
    activation='swish',
    final_activation='sigmoid',
    normalization='group',
    upsampling_mode='bilinear',
    use_self_attention=True,
    attention_resolution=32,
    dropout_rate=0.1
)
```

### GAN Training (Adversarial)
```python
ConvDecoder(
    latent_dim=128,
    hidden_dims=[512, 256, 128, 64],
    activation='relu',
    final_activation='tanh',  # For [-1,1] normalized images
    normalization='none',
    use_spectral_norm=True,
    upsampling_mode='transpose'
)
```

## Performance Tips

1. **Start Simple**: Use basic configuration, then optimize
2. **Match Preprocessing**: Ensure final_activation matches your image normalization
3. **Batch Size**: Use batch norm for large batches, group norm for small batches
4. **Memory vs Quality**: Self-attention improves quality but uses more memory
5. **Stability**: Use spectral norm if training is unstable
6. **Speed vs Quality**: Transpose convolutions are faster but may create artifacts

## Common Issues and Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| Poor reconstruction | `latent_dim` too small | Increase latent dimension |
| Checkerboard artifacts | Transpose convolutions | Use `upsampling_mode='bilinear'` |
| Training instability | No normalization | Add batch/group normalization |
| Overfitting | No regularization | Add dropout (0.1-0.2) |
| Slow convergence | Poor activation | Try ReLU, GELU, or Swish |
| Output range issues | Wrong final activation | Match final_activation to data range |
| Vanishing gradients | Too deep, bad activation | Use ReLU/GELU, add normalization |
| Memory issues | Self-attention on large images | Reduce attention_resolution |

## Hyperparameter Search Strategy

1. **Phase 1**: Fix architecture (`latent_dim`, `hidden_dims`, `output_size`)
2. **Phase 2**: Optimize activations (`activation`, `final_activation`)
3. **Phase 3**: Tune regularization (`normalization`, `dropout_rate`)
4. **Phase 4**: Fine-tune upsampling (`upsampling_mode`, `kernel_size`)
5. **Phase 5**: Add advanced features (attention, spectral norm) if needed

This systematic approach prevents hyperparameter explosion while ensuring good performance.