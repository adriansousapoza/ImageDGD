import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Tuple
import math

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
                 use_batch_norm: bool = True,
                 activation: str = 'leaky_relu',
                 final_activation: str = 'sigmoid',
                 dropout_rate: float = 0.0,
                 **kwargs) -> None:
        super(ConvDecoder, self).__init__()
        
        # Define available activation functions
        self.activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
        }
        
        self.final_activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'none': nn.Identity(),
        }
        
        # Validate activation choice
        if activation not in self.activations:
            raise ValueError(f"Invalid activation function: '{activation}'. "
                            f"Available options: {list(self.activations.keys())}")
            
        # Validate final activation choice
        if final_activation not in self.final_activations:
            raise ValueError(f"Invalid final activation function: '{final_activation}'. "
                            f"Available options: {list(self.final_activations.keys())}")
        
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
        
        # Reverse hidden_dims to go from smallest (latent) to largest (image)
        hidden_dims.reverse()
        
        # Get the selected activation functions
        activation_fn = self.activations[activation]
        final_activation_fn = self.final_activations[final_activation]
        
        # Initial linear projection from latent space to spatial dimensions
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * init_size[0] * init_size[1])
        
        # Build decoder layers
        modules = []
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]) if use_batch_norm else nn.Identity(),
                    activation_fn,
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                )
            )
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]) if use_batch_norm else nn.Identity(),
            activation_fn,
            nn.Conv2d(hidden_dims[-1], out_channels=output_channels, kernel_size=3, padding=1),
            final_activation_fn
        )
        
        # Add upsampling layer if needed to match exact output dimensions
        current_h = init_size[0] * (2 ** (len(modules) + 1))
        current_w = init_size[1] * (2 ** (len(modules) + 1))
        
        self.needs_final_upsample = (current_h != output_size[0] or current_w != output_size[1])
        if self.needs_final_upsample:
            self.final_upsample = nn.Upsample(size=output_size, mode='bilinear', align_corners=False)
        
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
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        
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
    
    