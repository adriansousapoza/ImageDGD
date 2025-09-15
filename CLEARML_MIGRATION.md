# ClearML Migration & Optimizer Enhancement Guide

This repository has been migrated from MLflow to ClearML for experiment tracking and enhanced with flexible optimizer support.

## What Changed

### Dependencies
- Replaced `mlflow>=2.8.0` with `clearml>=1.13.0` in `requirements.txt`
- Added optional optimizer dependencies: `lion-pytorch>=0.0.6` and `sophia-opt>=0.2.1`

### Code Changes
- Updated `src/training/trainer.py`:
  - Replaced `import mlflow` with `from clearml import Task`
  - Updated `DGDTrainer` class to use ClearML instead of MLflow
  - Added automatic ClearML task initialization in the trainer constructor
  - Replaced `mlflow.log_metric()` with ClearML's `logger.report_scalar()`
  - Replaced `mlflow.log_params()` with ClearML's `task.set_parameters()`
  - **NEW**: Enhanced optimizer system supporting all PyTorch optimizers plus Lion and Sophia

### Optimizer Enhancement
- **Flexible Optimizer Selection**: All PyTorch optimizers supported as hyperparameters
- **Additional Optimizers**: Lion and Sophia optimizers for high performance
- **Smart Parameter Filtering**: Automatically handles optimizer-specific parameters
- **Case Insensitive**: Optimizer names work regardless of case
- **Mixed Optimizers**: Different optimizers for decoder and representation layers

### Configuration Files
- Updated `.gitignore` to exclude ClearML directories instead of MLflow

## Setting Up ClearML

1. **Install ClearML:**
   ```bash
   pip install clearml
   ```

2. **Configure ClearML:**
   ```bash
   clearml-init
   ```
   This will prompt you to enter your ClearML credentials and server details.

3. **Install Optional Optimizers (recommended):**
   ```bash
   pip install lion-pytorch sophia-opt
   ```

4. **Verify Installation:**
   Run the test scripts to verify everything works:
   ```bash
   python test_clearml_integration.py
   python test_optimizers.py
   ```

## Enhanced Optimizer Usage

### Supported Optimizers

**PyTorch Built-in:**
- SGD, Adam, AdamW, Adagrad, Adadelta, Adamax, ASGD, RMSprop, Rprop, LBFGS, NAdam, RAdam, SparseAdam

**High-Performance Additional:**
- **Lion**: Evolved sign momentum (often outperforms Adam with lower memory)
- **Sophia**: Second-order clipped stochastic optimization (faster convergence)

### Configuration Examples

**AdamW (Default with No Weight Decay for Decoder):**
```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.0        # No weight decay for decoder (recommended)
    representation:
      type: "AdamW"
      lr: 0.01
      weight_decay: 0.01       # Weight decay for representations
```

**Lion Optimizer (High Performance):**
```yaml
training:
  optimizer:
    decoder:
      type: "Lion"
      lr: 0.0001               # Lion uses lower learning rates
      weight_decay: 0.0        # No weight decay for decoder
    representation:
      type: "Lion"
      lr: 0.001
      weight_decay: 0.01
```

**Mixed Optimizers (Conservative Decoder, Aggressive Representations):**
```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"            # Conservative for generative decoder
      lr: 0.001
      weight_decay: 0.0        # No weight decay for decoder
    representation:
      type: "Lion"             # High-performance for embeddings
      lr: 0.001
      weight_decay: 0.01       # Regularization for representations
```

**Lion Optimizer (High Performance):**
```yaml
training:
  optimizer:
    decoder:
      type: "Lion"
      lr: 0.0001  # Lion uses lower learning rates
      weight_decay: 0.01
    representation:
      type: "Lion"
      lr: 0.001
      weight_decay: 0.01
```

**SGD with Momentum:**
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

**Mixed Optimizers:**
```yaml
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.01
    representation:
      type: "Lion"  # Different optimizer for representations
      lr: 0.001
      weight_decay: 0.01
```

## Using ClearML with DGDTrainer

The `DGDTrainer` class now automatically initializes a ClearML task. You can either:

1. **Let the trainer create a task automatically:**
   ```python
   trainer = DGDTrainer(config, device)
   # Task will be created with project_name="ImageDGD", task_name="DGD Training"
   ```

2. **Create your own task first:**
   ```python
   from clearml import Task
   task = Task.init(project_name="MyProject", task_name="MyExperiment")
   trainer = DGDTrainer(config, device)
   # Trainer will use the existing task
   ```

## Benefits

### ClearML Benefits
- **Better Experiment Management**: Comprehensive web UI for experiment tracking
- **Model Management**: Built-in model registry and versioning
- **Data Management**: Automatic dataset versioning and lineage tracking
- **Resource Monitoring**: Automatic tracking of system resources, GPU usage, etc.
- **Collaboration**: Better team collaboration features
- **Pipelines**: Support for ML pipelines and automation

### Optimizer Enhancement Benefits
- **Hyperparameter Optimization**: Easy to change optimizers for hyperparameter tuning
- **Performance**: Access to state-of-the-art optimizers like Lion and Sophia
- **Flexibility**: Mix different optimizers for different model components
- **Robustness**: Automatic parameter validation and filtering
- **Ease of Use**: Simple configuration changes to try different optimizers

## Migration Notes

- All existing functionality remains the same
- Metrics and parameters are now logged to ClearML instead of MLflow
- Configuration objects are automatically connected to ClearML tasks
- The notebook (`DGD_fmnist.ipynb`) doesn't require changes as it doesn't use the trainer class
- **NEW**: Optimizer configuration is now much more flexible and powerful

## Performance Tips

1. **Lion Optimizer**: Often achieves better performance than Adam with 50% less memory usage
2. **Sophia Optimizer**: Can converge 2x faster on some problems due to second-order information
3. **Weight Decay Guidelines**:
   - **Decoder**: Use `weight_decay: 0.0` (no weight decay) - generative models need full capacity
   - **Representations**: Use `weight_decay: 0.01` - embeddings benefit from regularization
4. **Learning Rate Guidelines**:
   - Lion: 3-10x lower than Adam (e.g., 0.0001 vs 0.001)
   - Sophia: Similar to Adam (0.001-0.01)
   - AdamW: 0.001-0.01
   - SGD: 0.01-0.1
   - Representations: Often need 5-20x higher LR than decoder

## Troubleshooting

### ClearML Issues
1. Make sure ClearML is properly configured with `clearml-init`
2. Check that your ClearML server is accessible
3. Run the test script to verify the integration

### Optimizer Issues
1. **Import Errors**: Install optional optimizers with `pip install lion-pytorch sophia-opt`
2. **Parameter Errors**: The system will warn about unsupported parameters but continue training
3. **Performance Issues**: Try different learning rates (Lion typically needs lower rates)

## Documentation

- **ClearML**: https://clear.ml/docs/latest/docs/
- **Optimizer Guide**: See `OPTIMIZER_GUIDE.md` for detailed optimizer configuration
- **Test Scripts**: `test_clearml_integration.py` and `test_optimizers.py` for verification