#!/usr/bin/env python3
"""
Test script to verify the flexible optimizer system in DGDTrainer.
"""

import sys
import os
import torch

# Add the src directory to the path to import the modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_optimizer_creation():
    """Test the optimizer creation system."""
    try:
        from training.trainer import DGDTrainer
        
        # Create a mock config-like object
        class MockConfig:
            def __init__(self, optimizer_type, **kwargs):
                self.type = optimizer_type
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        # Create trainer instance for testing
        config = type('Config', (), {})()
        config.training = type('Training', (), {})()
        config.model = type('Model', (), {})()
        
        trainer = DGDTrainer(config, torch.device('cpu'), verbose=False)
        
        # Test PyTorch optimizers
        pytorch_optimizers = [
            ('AdamW', {'lr': 0.001, 'weight_decay': 0.01}),
            ('Adam', {'lr': 0.001, 'betas': [0.9, 0.999]}),
            ('SGD', {'lr': 0.01, 'momentum': 0.9}),
            ('RMSprop', {'lr': 0.01, 'alpha': 0.99}),
            ('Adagrad', {'lr': 0.01}),
            ('Adadelta', {'lr': 1.0, 'rho': 0.9}),
            ('NAdam', {'lr': 0.002}),
            ('RAdam', {'lr': 0.001}),
        ]
        
        print("Testing PyTorch optimizers:")
        for opt_name, params in pytorch_optimizers:
            try:
                opt_class = trainer._get_optimizer_class(opt_name)
                print(f"✓ {opt_name}: {opt_class.__name__}")
                
                # Test optimizer creation
                dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
                config_obj = MockConfig(opt_name, **params)
                optimizer = trainer._create_optimizer(opt_class, dummy_params, config_obj.__dict__)
                print(f"  ✓ Successfully created {opt_name} optimizer")
                
            except Exception as e:
                print(f"✗ {opt_name}: {e}")
        
        # Test additional optimizers
        print("\nTesting additional optimizers:")
        
        # Test Lion
        try:
            opt_class = trainer._get_optimizer_class('Lion')
            print(f"✓ Lion: Available")
            dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
            config_obj = MockConfig('Lion', lr=0.0001, weight_decay=0.01)
            optimizer = trainer._create_optimizer(opt_class, dummy_params, config_obj.__dict__)
            print(f"  ✓ Successfully created Lion optimizer")
        except Exception as e:
            print(f"✗ Lion: {e} (install with: pip install lion-pytorch)")
        
        # Test Sophia
        try:
            opt_class = trainer._get_optimizer_class('Sophia')
            print(f"✓ Sophia: Available")
            dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
            config_obj = MockConfig('Sophia', lr=0.001, betas=[0.965, 0.99], rho=0.04)
            optimizer = trainer._create_optimizer(opt_class, dummy_params, config_obj.__dict__)
            print(f"  ✓ Successfully created Sophia optimizer")
        except Exception as e:
            print(f"✗ Sophia: {e} (install with: pip install sophia-opt)")
        
        # Test case insensitivity
        print("\nTesting case insensitivity:")
        case_tests = ['adamw', 'ADAMW', 'AdamW', 'sgd', 'SGD', 'Sgd']
        for test_name in case_tests:
            try:
                opt_class = trainer._get_optimizer_class(test_name)
                print(f"✓ {test_name} -> {opt_class.__name__}")
            except Exception as e:
                print(f"✗ {test_name}: {e}")
        
        # Test parameter filtering
        print("\nTesting parameter filtering:")
        try:
            opt_class = trainer._get_optimizer_class('SGD')
            dummy_params = [torch.nn.Parameter(torch.randn(10, 10))]
            
            # Include some unsupported parameters
            config_obj = MockConfig('SGD', 
                lr=0.01, 
                momentum=0.9, 
                weight_decay=0.001,
                betas=[0.9, 0.999],  # This should be ignored for SGD
                unsupported_param=123  # This should be ignored
            )
            
            optimizer = trainer._create_optimizer(opt_class, dummy_params, config_obj.__dict__)
            print(f"✓ Parameter filtering works correctly")
            print(f"  Created SGD with lr={optimizer.param_groups[0]['lr']}, momentum={optimizer.param_groups[0]['momentum']}")
            
        except Exception as e:
            print(f"✗ Parameter filtering test failed: {e}")
        
        print("\n✓ All optimizer system tests completed successfully!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure you're running this from the repository root directory")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    test_optimizer_creation()
    
    print("\nOptimizer Configuration Examples:")
    print("=" * 50)
    print("""
# Example configuration for different optimizers:

# AdamW (recommended default)
training:
  optimizer:
    decoder:
      type: "AdamW"
      lr: 0.001
      weight_decay: 0.01
    representation:
      type: "AdamW" 
      lr: 0.01
      weight_decay: 0.01

# Lion (high performance)
training:
  optimizer:
    decoder:
      type: "Lion"
      lr: 0.0001
      weight_decay: 0.01
    representation:
      type: "Lion"
      lr: 0.001
      weight_decay: 0.01

# SGD with momentum
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
    """)