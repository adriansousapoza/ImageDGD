#!/usr/bin/env python3
"""
Test script for the enhanced ConvDecoder with new hyperparameters.
Demonstrates the flexibility and new features of the decoder architecture.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.dgd import ConvDecoder

def test_basic_decoder():
    """Test basic decoder functionality."""
    print("Testing basic ConvDecoder...")
    
    decoder = ConvDecoder(
        latent_dim=64,
        hidden_dims=[256, 128, 64],
        output_size=(28, 28),
        activation='relu',
        final_activation='sigmoid'
    )
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 64)
    output = decoder(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: {(batch_size, 1, 28, 28)}")
    
    assert output.shape == (batch_size, 1, 28, 28), f"Expected {(batch_size, 1, 28, 28)}, got {output.shape}"
    assert 0 <= output.min() and output.max() <= 1, "Output should be in [0,1] range with sigmoid"
    
    print("✓ Basic decoder test passed!\n")

def test_different_activations():
    """Test different activation functions."""
    print("Testing different activation functions...")
    
    activations_to_test = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu']
    
    for activation in activations_to_test:
        try:
            decoder = ConvDecoder(
                latent_dim=32,
                hidden_dims=[128, 64],
                output_size=(28, 28),
                activation=activation,
                final_activation='sigmoid'
            )
            
            z = torch.randn(2, 32)
            output = decoder(z)
            
            print(f"✓ {activation} activation works correctly")
        except Exception as e:
            print(f"✗ {activation} activation failed: {e}")
    
    print()

def test_different_normalizations():
    """Test different normalization options."""
    print("Testing different normalization options...")
    
    normalizations = ['batch', 'group', 'layer', 'instance', 'none']
    
    for norm in normalizations:
        try:
            decoder = ConvDecoder(
                latent_dim=32,
                hidden_dims=[128, 64],
                output_size=(28, 28),
                normalization=norm
            )
            
            z = torch.randn(2, 32)
            output = decoder(z)
            
            print(f"✓ {norm} normalization works correctly")
        except Exception as e:
            print(f"✗ {norm} normalization failed: {e}")
    
    print()

def test_upsampling_modes():
    """Test different upsampling modes."""
    print("Testing different upsampling modes...")
    
    upsampling_modes = ['transpose', 'bilinear', 'nearest', 'bicubic']
    
    for mode in upsampling_modes:
        try:
            decoder = ConvDecoder(
                latent_dim=32,
                hidden_dims=[128, 64],
                output_size=(28, 28),
                upsampling_mode=mode
            )
            
            z = torch.randn(2, 32)
            output = decoder(z)
            
            print(f"✓ {mode} upsampling works correctly")
        except Exception as e:
            print(f"✗ {mode} upsampling failed: {e}")
    
    print()

def test_advanced_features():
    """Test advanced features like self-attention and spectral normalization."""
    print("Testing advanced features...")
    
    # Test self-attention
    try:
        decoder = ConvDecoder(
            latent_dim=64,
            hidden_dims=[256, 128, 64],
            output_size=(32, 32),  # Larger size for attention
            use_self_attention=True,
            attention_resolution=16
        )
        
        z = torch.randn(2, 64)
        output = decoder(z)
        
        print("✓ Self-attention works correctly")
    except Exception as e:
        print(f"✗ Self-attention failed: {e}")
    
    # Test spectral normalization
    try:
        decoder = ConvDecoder(
            latent_dim=32,
            hidden_dims=[128, 64],
            output_size=(28, 28),
            use_spectral_norm=True
        )
        
        z = torch.randn(2, 32)
        output = decoder(z)
        
        print("✓ Spectral normalization works correctly")
    except Exception as e:
        print(f"✗ Spectral normalization failed: {e}")
    
    print()

def test_parameter_count():
    """Compare parameter counts for different configurations."""
    print("Comparing parameter counts...")
    
    configs = [
        {"name": "Basic", "latent_dim": 64, "hidden_dims": [256, 128, 64]},
        {"name": "Deep", "latent_dim": 64, "hidden_dims": [512, 256, 128, 64, 32]},
        {"name": "Wide", "latent_dim": 128, "hidden_dims": [512, 256, 128]},
        {"name": "With Attention", "latent_dim": 64, "hidden_dims": [256, 128, 64], "use_self_attention": True},
    ]
    
    for config in configs:
        name = config.pop("name")
        decoder = ConvDecoder(output_size=(28, 28), **config)
        
        total_params = sum(p.numel() for p in decoder.parameters())
        trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        
        print(f"{name:15}: {total_params:,} total, {trainable_params:,} trainable")
    
    print()

def test_backward_compatibility():
    """Test backward compatibility with old use_batch_norm parameter."""
    print("Testing backward compatibility...")
    
    try:
        # Old style initialization
        decoder = ConvDecoder(
            latent_dim=32,
            hidden_dims=[128, 64],
            output_size=(28, 28),
            use_batch_norm=True  # Old parameter
        )
        
        z = torch.randn(2, 32)
        output = decoder(z)
        
        print("✓ Backward compatibility with use_batch_norm works")
    except Exception as e:
        print(f"✗ Backward compatibility failed: {e}")
    
    print()

def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing edge cases...")
    
    # Test invalid activation
    try:
        ConvDecoder(latent_dim=32, hidden_dims=[64], activation='invalid_activation')
        print("✗ Should have raised error for invalid activation")
    except ValueError:
        print("✓ Correctly raised error for invalid activation")
    
    # Test invalid normalization
    try:
        ConvDecoder(latent_dim=32, hidden_dims=[64], normalization='invalid_norm')
        print("✗ Should have raised error for invalid normalization")
    except ValueError:
        print("✓ Correctly raised error for invalid normalization")
    
    # Test invalid upsampling mode
    try:
        ConvDecoder(latent_dim=32, hidden_dims=[64], upsampling_mode='invalid_mode')
        print("✗ Should have raised error for invalid upsampling mode")
    except ValueError:
        print("✓ Correctly raised error for invalid upsampling mode")
    
    print()

def main():
    """Run all tests."""
    print("Enhanced ConvDecoder Test Suite")
    print("=" * 50)
    
    test_basic_decoder()
    test_different_activations()
    test_different_normalizations()
    test_upsampling_modes()
    test_advanced_features()
    test_parameter_count()
    test_backward_compatibility()
    test_edge_cases()
    
    print("All tests completed!")

if __name__ == "__main__":
    main()