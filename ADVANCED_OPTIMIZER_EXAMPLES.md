# Advanced Optimizer Configuration Examples

This guide shows how to configure different optimizers and parameters for decoder vs. representation layers, including the common case of using no weight decay for the decoder.

## Separate Optimizer Configuration

Your DGDTrainer supports completely independent optimizer configuration for:
- **Decoder**: The generative decoder network
- **Representations**: The learnable representation embeddings (both train and test)

## Common Use Cases

### 1. No Weight Decay for Decoder (Recommended)

Many deep learning practitioners find that weight decay can hurt generative models like decoders. Here's how to configure no weight decay for the decoder while using weight decay for representations:

```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0        # No weight decay for decoder
      betas: [0.9, 0.999]
      eps: 1e-08
    representation:
      type: "AdamW"
      lr: 0.01
      weight_decay: 0.01       # Weight decay for representations
      betas: [0.9, 0.999]
      eps: 1e-08
```

### 2. Different Optimizers for Each Component

You can use completely different optimizers. For example, conservative AdamW for the decoder and aggressive Lion for representations:

```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0        # No weight decay
      betas: [0.9, 0.999]
    representation:
      type: "Lion"
      lr: 0.001
      weight_decay: 0.01       # With weight decay
```

### 3. SGD for Decoder, Adam for Representations

```yaml
training:
  optimizer:
    decoder:
      type: "SGD"
      lr: 0.01
      momentum: 0.9
      weight_decay: 0.0        # No weight decay for decoder
      nesterov: true
    representation:
      type: "Adam"
      lr: 0.01
      betas: [0.9, 0.999]
      weight_decay: 0.001      # Light weight decay for representations
      eps: 1e-08
```

### 4. High-Performance Configuration

Using state-of-the-art optimizers for maximum performance:

```yaml
training:
  optimizer:
    decoder:
      type: "Sophia"           # Second-order optimizer for decoder
      lr: 0.001
      betas: [0.965, 0.99]
      rho: 0.04
      weight_decay: 0.0        # No weight decay
    representation:
      type: "Lion"             # Evolved sign momentum for representations
      lr: 0.001
      weight_decay: 0.01
```

### 5. Different Learning Rates

Often representations need higher learning rates than the decoder:

```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.0005               # Lower LR for stable decoder training
      weight_decay: 0.0
    representation:
      type: "AdamW" 
      lr: 0.02                 # Higher LR for faster representation learning
      weight_decay: 0.01
```

## Advanced Configurations

### Conservative Decoder, Aggressive Representations

```yaml
training:
  optimizer:
    decoder:
      type: "Adam"             # Conservative Adam
      lr: 0.0001
      betas: [0.9, 0.999]
      weight_decay: 0.0
      eps: 1e-08
      amsgrad: false
    representation:
      type: "Lion"             # Aggressive Lion
      lr: 0.005
      weight_decay: 0.05
```

### Research Configuration with RMSprop

```yaml
training:
  optimizer:
    decoder:
      type: "RMSprop"
      lr: 0.001
      alpha: 0.99
      eps: 1e-08
      weight_decay: 0.0        # No weight decay
      momentum: 0.0
      centered: false
    representation:
      type: "RMSprop"
      lr: 0.01
      alpha: 0.95
      eps: 1e-08
      weight_decay: 0.001      # Light weight decay
      momentum: 0.1
      centered: true
```

## Why No Weight Decay for Decoder?

**Theoretical Reasons:**
1. **Generative Models**: Decoders are generative models that need to learn complex data distributions
2. **Capacity**: Weight decay reduces model capacity, which can hurt generation quality
3. **Overfitting**: Generative models typically don't overfit in the same way as discriminative models

**Empirical Evidence:**
- Many successful generative models (GANs, VAEs, diffusion models) use little to no weight decay
- Weight decay can lead to blurry or low-quality generated samples
- Representations (embeddings) benefit more from regularization than decoder weights

## Parameter Isolation

Each optimizer configuration is completely independent:

```yaml
training:
  optimizer:
    decoder:
      # Only affects the ConvDecoder parameters
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0
      betas: [0.9, 0.999]
      eps: 1e-08
      
    representation:  
      # Only affects RepresentationLayer parameters (both train and test)
      type: "Lion"
      lr: 0.001
      weight_decay: 0.01
```

## Validation and Debugging

The trainer will print warnings if you specify unsupported parameters:

```yaml
training:
  optimizer:
    decoder:
      type: "SGD"
      lr: 0.01
      momentum: 0.9
      betas: [0.9, 0.999]      # WARNING: Not supported by SGD, will be ignored
      weight_decay: 0.0
```

## Best Practices

1. **Start Simple**: Begin with AdamW for both, no weight decay for decoder
2. **Tune Learning Rates**: Representations often need higher LR (5-20x decoder LR)
3. **Weight Decay**: Start with 0.0 for decoder, 0.01 for representations
4. **Experiment**: Try Lion for representations, keep AdamW for decoder
5. **Monitor**: Use ClearML to track which combinations work best

## Complete Example Configuration

```yaml
# Recommended starting configuration
training:
  epochs: 200
  first_epoch_gmm: 50
  refit_gmm_interval: 10
  lambda_gmm: 1.0
  
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0        # Critical: No weight decay for decoder
      betas: [0.9, 0.999]
      eps: 1e-08
      
    representation:
      type: "Lion"             # High-performance optimizer
      lr: 0.01                 # Higher LR for representations
      weight_decay: 0.01       # Regularization for embeddings
      
  logging:
    log_interval: 10
    plot_interval: 50
    save_figures: true
```