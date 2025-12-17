import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple
import math


class DGD(nn.Module):
    """
    Complete model that combines a Decoder with a RepresentationLayer and GMM.
    """
    def __init__(self, decoder, rep_layer=None, gmm=None):
        super(DGD, self).__init__()
        self.decoder = decoder
        self.rep_layer = rep_layer
        self.gmm = gmm
        
    def forward(self, indices=None, z=None):
        """
        Forward pass through the model.
        Either provide indices to use representations from the rep_layer,
        or provide z directly to bypass the rep_layer.
        """
        if z is None and indices is not None and self.rep_layer is not None:
            z = self.rep_layer(indices)
        elif z is None and indices is None:
            raise ValueError("Either indices or z must be provided")
            
        return self.decoder(z)
    
    def sample_from_gmm(self, n_samples=1):
        """
        Sample from the GMM and generate images.
        """
        if self.gmm is None:
            raise ValueError("GMM is not set")
            
        z, _ = self.gmm.sample(n_samples)
        return self.decoder(z)
    
    def loss_function(self, x, indices=None, z=None, gmm_weight=1.0):
        """
        Compute the loss function.
        """
        if z is None and indices is not None and self.rep_layer is not None:
            z = self.rep_layer(indices)
        elif z is None and indices is None:
            raise ValueError("Either indices or z must be provided")
            
        # Reconstruction
        x_recon = self.decoder(z)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # GMM loss
        gmm_loss = 0
        if self.gmm is not None:
            gmm_loss = -gmm_weight * torch.sum(self.gmm.score_samples(z))
            
        # Total loss
        total_loss = recon_loss + gmm_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'gmm_loss': gmm_loss,
            'recon': x_recon
        }


class BaseDecoder(nn.Module):
    """
    Base decoder class for encoder-free framework
    """
    def __init__(self) -> None:
        super(BaseDecoder, self).__init__()

    @abstractmethod
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, z: Tensor, **kwargs) -> Tensor:
        """
        Forward pass through the decoder.
        :param z: (Tensor) Input tensor with latent representations [B x D]
        :return: (Tensor) Output tensor with reconstructed images
        """
        pass

    def sample(self, num_samples: int, latent_dim: int, device: torch.device, **kwargs) -> Tensor:
        """
        Samples from a standard normal distribution and transforms
        to image space using the decoder.
        :param num_samples: (Int) Number of samples
        :param latent_dim: (Int) Dimensionality of latent space
        :param device: Device to run the model on
        :return: (Tensor) [B x C x H x W]
        """
        z = torch.randn(num_samples, latent_dim).to(device)
        return self.decode(z)
    
    def generate(self, z: Tensor, **kwargs) -> Tensor:
        """
        Given a latent vector z, returns the reconstructed image
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decode(z)


class ConvDecoder(BaseDecoder):
    """
    Convolutional decoder that follows a VAE-like structure
    but works with external representation layers.
    """
    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List = None,
                 output_channels: int = 1,
                 output_size: Tuple[int, int] = (28, 28),
                 init_size: Tuple[int, int] = None,
                 kernel_size: int = 3,
                 stride: int = 2,
                 padding: int = 1,
                 output_padding: int = 1,
                 normalization: str = 'batch',
                 use_batch_norm: bool = True,  # Deprecated, use normalization
                 activation: str = 'leaky_relu',
                 final_activation: str = 'sigmoid',
                 dropout_rate: float = 0.0,
                 upsampling_mode: str = 'transpose',
                 **kwargs) -> None:
        super(ConvDecoder, self).__init__()
        
        # Define all available PyTorch activation functions
        self.activations = {
            # Standard activations
            'relu': nn.ReLU(),
            'relu6': nn.ReLU6(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'prelu': nn.PReLU(),
            'rrelu': nn.RReLU(),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'celu': nn.CELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),  # SiLU is Swish
            'silu': nn.SiLU(),
            'mish': nn.Mish(),
            'hardswish': nn.Hardswish(),
            'hardsigmoid': nn.Hardsigmoid(),
            'hardtanh': nn.Hardtanh(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'logsigmoid': nn.LogSigmoid(),
            'softplus': nn.Softplus(),
            'softshrink': nn.Softshrink(),
            'softsign': nn.Softsign(),
            'tanhshrink': nn.Tanhshrink(),
            'hardshrink': nn.Hardshrink(),
            'threshold': nn.Threshold(0.1, 0.0),
            'glu': nn.GLU(),
            # Special
            'identity': nn.Identity(),
            'none': nn.Identity(),
        }
        
        self.final_activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'hardsigmoid': nn.Hardsigmoid(),
            'hardtanh': nn.Hardtanh(),
            'softsign': nn.Softsign(),
            'identity': nn.Identity(),
            'none': nn.Identity(),
        }
        
        # Define normalization options
        self.normalizations = {
            'batch': lambda channels: nn.BatchNorm2d(channels),
            'group': lambda channels: nn.GroupNorm(min(32, channels), channels),
            'layer': lambda channels: nn.GroupNorm(1, channels),  # LayerNorm for 2D
            'instance': lambda channels: nn.InstanceNorm2d(channels),
            'spectral': lambda channels: nn.Identity(),  # Handled separately
            'none': lambda channels: nn.Identity(),
        }
        
        # Validate activation choice
        if activation not in self.activations:
            raise ValueError(f"Invalid activation function: '{activation}'. "
                            f"Available options: {list(self.activations.keys())}")
            
        # Validate final activation choice
        if final_activation not in self.final_activations:
            raise ValueError(f"Invalid final activation function: '{final_activation}'. "
                            f"Available options: {list(self.final_activations.keys())}")
        
        # Validate normalization choice (backward compatibility)
        if normalization == 'batch' and not use_batch_norm:
            normalization = 'none'
        elif use_batch_norm and normalization != 'batch':
            print(f"Warning: use_batch_norm=True but normalization='{normalization}'. Using normalization setting.")
        
        if normalization not in self.normalizations:
            raise ValueError(f"Invalid normalization: '{normalization}'. "
                            f"Available options: {list(self.normalizations.keys())}")
        
        # Validate upsampling mode
        valid_upsampling = ['transpose', 'nearest', 'bilinear', 'bicubic']
        if upsampling_mode not in valid_upsampling:
            raise ValueError(f"Invalid upsampling mode: '{upsampling_mode}'. "
                            f"Available options: {valid_upsampling}")
        
        # Store configuration
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.normalization = normalization
        self.upsampling_mode = upsampling_mode
        
        # Convert Hydra ListConfig objects to regular Python tuples/lists
        # This is necessary because PyTorch layers expect native Python types
        if hasattr(output_size, '_content'):  # Check if it's a ListConfig
            output_size = tuple(output_size)
        if init_size is not None and hasattr(init_size, '_content'):  # Check if it's a ListConfig
            init_size = tuple(init_size)
        if hidden_dims is not None and hasattr(hidden_dims, '_content'):  # Check if it's a ListConfig
            hidden_dims = list(hidden_dims)
        
        self.latent_dim = latent_dim
        
        # Set default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]
        
        # Determine the initial size to start reconstruction
        if init_size is None:
            # Default to 1/8 of the output size, but at least 2x2
            init_size = (max(2, output_size[0] // 8), max(2, output_size[1] // 8))
        
        self.init_size = init_size
        
        # Calculate number of upsample operations needed
        h_upsamples = int(math.log2(output_size[0] / init_size[0]))
        w_upsamples = int(math.log2(output_size[1] / init_size[1]))
        num_upsamples = max(h_upsamples, w_upsamples)
        
        # Make sure we have enough hidden_dims for the required upsampling
        if len(hidden_dims) < num_upsamples + 1:
            # Extend hidden_dims if needed
            last_size = hidden_dims[-1]
            for _ in range(num_upsamples + 1 - len(hidden_dims)):
                hidden_dims.append(last_size // 2 if last_size > 8 else last_size)
        
        # hidden_dims should be specified high→low (e.g., [128, 64, 32])
        # This gives high capacity at low resolution (semantic) → low capacity at high resolution (spatial)
        # No reversal needed - use as-is from config
        
        # Get the selected activation functions
        activation_fn = self.activations[activation]
        final_activation_fn = self.final_activations[final_activation]
        
        # Initial linear projection from latent space to spatial dimensions
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * init_size[0] * init_size[1])
        
        # Build decoder layers
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            # Build convolution layer
            if upsampling_mode == 'transpose':
                # Use transposed convolution for upsampling
                conv_layer = nn.ConvTranspose2d(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                )
            else:
                # Use regular convolution with separate upsampling
                if upsampling_mode == 'nearest':
                    upsample_layer = nn.Upsample(scale_factor=2, mode=upsampling_mode)
                else:
                    upsample_layer = nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=False)
                conv_layer = nn.Conv2d(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    kernel_size=kernel_size,
                    stride=1,  # No stride with separate upsampling
                    padding=padding,
                )
            
            # Build layer sequence
            layer_modules = []
            
            # Add upsampling first if using separate upsampling
            if upsampling_mode != 'transpose':
                layer_modules.append(upsample_layer)
            
            layer_modules.append(conv_layer)
            
            # Add normalization
            if normalization == 'batch':
                layer_modules.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            elif normalization == 'layer':
                layer_modules.append(nn.GroupNorm(1, hidden_dims[i + 1]))
            elif normalization == 'group':
                num_groups = min(8, hidden_dims[i + 1])
                layer_modules.append(nn.GroupNorm(num_groups, hidden_dims[i + 1]))
            elif normalization == 'instance':
                layer_modules.append(nn.InstanceNorm2d(hidden_dims[i + 1]))
            
            # Add activation
            layer_modules.append(activation_fn)
            
            # Add dropout
            if dropout_rate > 0:
                layer_modules.append(nn.Dropout(dropout_rate))
            
            modules.append(nn.Sequential(*layer_modules))
        
        # Final output layer
        final_modules = []
        
        if upsampling_mode == 'transpose':
            # Use transposed convolution for final upsampling
            final_conv1 = nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        else:
            # Use regular convolution with separate upsampling
            if upsampling_mode == 'nearest':
                final_modules.append(nn.Upsample(scale_factor=2, mode=upsampling_mode))
            else:
                final_modules.append(nn.Upsample(scale_factor=2, mode=upsampling_mode, align_corners=False))
            final_conv1 = nn.Conv2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
        
        final_modules.append(final_conv1)
        
        # Add normalization
        if normalization == 'batch':
            final_modules.append(nn.BatchNorm2d(hidden_dims[-1]))
        elif normalization == 'layer':
            final_modules.append(nn.GroupNorm(1, hidden_dims[-1]))
        elif normalization == 'group':
            num_groups = min(8, hidden_dims[-1])
            final_modules.append(nn.GroupNorm(num_groups, hidden_dims[-1]))
        elif normalization == 'instance':
            final_modules.append(nn.InstanceNorm2d(hidden_dims[-1]))
        
        # Add activation
        final_modules.append(activation_fn)
        
        # Final output convolution (no normalization, just output activation)
        final_conv2 = nn.Conv2d(hidden_dims[-1], out_channels=output_channels, kernel_size=3, padding=1)
        
        final_modules.extend([final_conv2, final_activation_fn])
        
        self.final_layer = nn.Sequential(*final_modules)
        
        # Add upsampling layer if needed to match exact output dimensions
        current_h = init_size[0] * (2 ** (len(modules) + 1))
        current_w = init_size[1] * (2 ** (len(modules) + 1))
        
        self.needs_final_upsample = (current_h != output_size[0] or current_w != output_size[1])
        if self.needs_final_upsample:
            # Use the same upsampling mode as the rest of the network
            upsample_mode = upsampling_mode if upsampling_mode != 'transpose' else 'bilinear'
            if upsample_mode == 'nearest':
                self.final_upsample = nn.Upsample(size=output_size, mode=upsample_mode)
            else:
                self.final_upsample = nn.Upsample(size=output_size, mode=upsample_mode, align_corners=False)
        
        self.decoder = nn.Sequential(*modules)
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, result.shape[1] // (self.init_size[0] * self.init_size[1]), 
                            self.init_size[0], self.init_size[1])
        result = self.decoder(result)
        result = self.final_layer(result)
        
        if self.needs_final_upsample:
            result = self.final_upsample(result)
            
        return result
    
    def forward(self, z: Tensor, **kwargs) -> Tensor:
        """
        Forward pass through the decoder.
        :param z: (Tensor) Input tensor with latent representations [B x D]
        :return: (Tensor) Output tensor with reconstructed images
        """
        return self.decode(z)