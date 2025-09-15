# Optimizer Configuration Guide

The DGDTrainer now supports all PyTorch optimizers plus additional high-performance optimizers like Lion and Sophia. This guide shows how to configure different optimizers as hyperparameters.

## Supported Optimizers

### PyTorch Built-in Optimizers
- **SGD**: Stochastic Gradient Descent
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Adam with Weight Decay
- **Adagrad**: Adaptive Gradient Algorithm
- **Adadelta**: Adaptive Learning Rate Method
- **Adamax**: Variant of Adam based on infinity norm
- **ASGD**: Averaged Stochastic Gradient Descent
- **RMSprop**: Root Mean Square Propagation
- **Rprop**: Resilient Backpropagation
- **LBFGS**: Limited-memory BFGS
- **NAdam**: Adam with Nesterov momentum
- **RAdam**: Rectified Adam
- **SparseAdam**: Lazy version of Adam for sparse tensors

### Additional High-Performance Optimizers
- **Lion**: Evolved sign momentum optimizer
- **Sophia**: Second-order clipped stochastic optimization

## Configuration Examples

### Independent Optimizer Configuration

**IMPORTANT**: The decoder and representation optimizers are completely independent. You can use different optimizers, learning rates, and parameters for each component.

### Basic Configuration (AdamW) - No Weight Decay for Decoder
```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0        # No weight decay for decoder (recommended)
    representation:
      type: "AdamW"
      lr: 0.01                 # Higher LR for representations
      weight_decay: 0.01       # Weight decay for representations
```

### Mixed Optimizers - Conservative Decoder, Aggressive Representations
```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"            # Conservative for decoder
      lr: 0.001
      weight_decay: 0.0        # No weight decay
    representation:
      type: "Lion"             # High-performance for representations
      lr: 0.001
      weight_decay: 0.01
```

### SGD with Momentum
```yaml
training:
  optimizer:
    decoder:
      type: "SGD"
      lr: 0.01
      momentum: 0.9
      weight_decay: 0.0001
    representation:
      type: "SGD"
      lr: 0.1
      momentum: 0.9
      weight_decay: 0.0001
```

### Adam Configuration
```yaml
training:
  optimizer:
    decoder:
      type: "Adam"
      lr: 0.001
      betas: [0.9, 0.999]
      eps: 1e-08
      weight_decay: 0
    representation:
      type: "Adam"
      lr: 0.01
      betas: [0.9, 0.999]
      eps: 1e-08
```

### Lion Optimizer (High Performance)
```yaml
training:
  optimizer:
    decoder:
      type: "Lion"
      lr: 0.0001  # Lion typically uses lower learning rates
      weight_decay: 0.01
    representation:
      type: "Lion"
      lr: 0.001
      weight_decay: 0.01
```

### Sophia Optimizer (Second-order)
```yaml
training:
  optimizer:
    decoder:
      type: "Sophia"
      lr: 0.001
      betas: [0.965, 0.99]
      rho: 0.04
      weight_decay: 0.01
    representation:
      type: "Sophia"
      lr: 0.01
      betas: [0.965, 0.99]
      rho: 0.04
      weight_decay: 0.01
```

### RMSprop Configuration
```yaml
training:
  optimizer:
    decoder:
      type: "RMSprop"
      lr: 0.01
      alpha: 0.99
      eps: 1e-08
      weight_decay: 0
      momentum: 0
    representation:
      type: "RMSprop"
      lr: 0.1
      alpha: 0.99
      eps: 1e-08
```

### Mixed Optimizers (Different for Each Component)
```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0        # No weight decay for decoder
    representation:
      type: "Lion"             # Use Lion for representations
      lr: 0.001
      weight_decay: 0.01
```

### SGD with Momentum - No Weight Decay for Decoder
```yaml
training:
  optimizer:
    decoder:
      type: "SGD"
      lr: 0.01
      momentum: 0.9
      weight_decay: 0.0        # No weight decay for decoder
    representation:
      type: "SGD"
      lr: 0.1
      momentum: 0.9
      weight_decay: 0.0001     # Light weight decay for representations
```

## Installation Notes

### Core Optimizers
All PyTorch optimizers are available by default when PyTorch is installed.

### Optional Optimizers
To use Lion and Sophia optimizers, install the additional packages:

```bash
# For Lion optimizer
pip install lion-pytorch

# For Sophia optimizer  
pip install sophia-opt
```

Or install all requirements including optional optimizers:
```bash
pip install -r requirements.txt
```

## Parameter Guidelines

### Learning Rate Recommendations
- **AdamW/Adam**: 0.001 - 0.01
- **SGD**: 0.01 - 0.1
- **Lion**: 0.0001 - 0.001 (typically 3-10x lower than Adam)
- **Sophia**: 0.001 - 0.01
- **RMSprop**: 0.001 - 0.01

### Weight Decay Recommendations
- **AdamW**: 0.01 - 0.1
- **SGD**: 0.0001 - 0.001
- **Lion**: 0.01 - 0.1
- **Sophia**: 0.01 - 0.1

## Advanced Usage

### Optimizer-Specific Parameters
The system automatically filters parameters based on what each optimizer supports. You can include any parameter supported by the optimizer:

```yaml
training:
  optimizer:
    decoder:
      type: "SGD"
      lr: 0.01
      momentum: 0.9
      dampening: 0
      weight_decay: 0.0001
      nesterov: true
    representation:
      type: "Adam"
      lr: 0.001
      betas: [0.9, 0.999]
      eps: 1e-08
      weight_decay: 0
      amsgrad: false
```

### Error Handling
If you specify a parameter not supported by an optimizer, the trainer will:
1. Print a warning message
2. Ignore the unsupported parameter
3. Continue with supported parameters only

### Case Insensitive
Optimizer names are case-insensitive:
- `"adamw"`, `"AdamW"`, `"ADAMW"` all work
- `"lion"`, `"Lion"`, `"LION"` all work

## Performance Tips

1. **Lion**: Often achieves better performance than Adam with lower memory usage
2. **Sophia**: Can converge faster on some problems due to second-order information
3. **AdamW**: Good general-purpose optimizer with weight decay
4. **SGD**: Sometimes achieves best final performance but may require more tuning

## Troubleshooting

### Import Errors
If you get import errors for Lion or Sophia:
```bash
pip install lion-pytorch sophia-opt
```

### Parameter Errors
Check the optimizer documentation for valid parameters. The trainer will warn about unsupported parameters but continue training.

### Performance Issues
- Try different learning rates (Lion typically needs lower rates)
- Adjust weight decay based on optimizer type
- Consider mixed optimizers for different model components