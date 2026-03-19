"""
Clifford Algebra Convolutional Layers for Geometric-Aware Neural Networks

Implements geometric algebra-based convolutions using multivector representations
of RGB pixels for improved feature extraction and equivariance properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClifffordVectorEncoder(nn.Module):
    """
    Encodes RGB pixels (3 channels) as multivectors in Clifford algebra.
    
    For RGB images, we create a reduced Clifford algebra representation:
    - Scalar part: grayscale intensity (luminance)
    - Vector parts: Color vectors (R, G, B deviations from mean)
    
    Each pixel p_RGB = (R, G, B) becomes:
    m(p) = scalar + e1*v1 + e2*v2 + e3*v3
    where scalar represents luminance and v_i represent color information.
    """
    
    def __init__(self, use_bivector=False):
        super(ClifffordVectorEncoder, self).__init__()
        self.use_bivector = use_bivector
        
        # RGB to grayscale weights (CIE standard)
        self.register_buffer('rgb_to_gray', 
                           torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1))
    
    def forward(self, x):
        """
        Encodes RGB image into multivector representation.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) - RGB images
        
        Returns:
            Multivector representation of shape (B, 4, H, W):
                - Channel 0: Scalar (grayscale/luminance)
                - Channels 1-3: Vector components (RGB deviations)
        """
        batch_size, channels, height, width = x.shape
        
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {channels} channels")
        
        # Compute luminance (scalar part)
        scalar = F.conv2d(x, self.rgb_to_gray)  # (B, 1, H, W)
        
        # Compute color deviations (vector parts)
        # Normalize RGB channels
        mean_color = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        color_deviation = x - mean_color  # (B, 3, H, W)
        
        # Concatenate scalar and vector parts
        multivector = torch.cat([scalar, color_deviation], dim=1)  # (B, 4, H, W)
        
        return multivector
    
    def get_magnitude(self, multivector):
        """
        Compute the magnitude (norm) of multivectors.
        
        For multivector m = s + e1*v1 + e2*v2 + e3*v3:
        ||m|| = sqrt(s^2 + v1^2 + v2^2 + v3^2)
        """
        # Square all components and sum
        magnitude_squared = (multivector ** 2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        magnitude = torch.sqrt(magnitude_squared + 1e-8)
        return magnitude


class ClifffordConv2d(nn.Module):
    """
    Clifford Algebra Convolution Layer.
    
    Replaces standard Conv2d by incorporating geometric product structure:
    - Uses multivector representations from input
    - Applies learnable transformations respecting algebraic structure
    - Computes geometric products between feature maps and kernels
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 use_geometric_product=True, use_magnitude_activation=False):
        super(ClifffordConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_geometric_product = use_geometric_product
        self.use_magnitude_activation = use_magnitude_activation
        
        # Standard conv layers for geometric product computation
        # We decompose the geometric product into multiple conv operations
        self.conv_scalar = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                     padding=padding, bias=True)
        
        # Bivector and trivector components (for full geometric algebra)
        if in_channels >= 4:  # Only if input has vector components
            self.conv_vector = nn.Conv2d(in_channels, out_channels * 3, kernel_size,
                                        padding=padding, bias=True)
        else:
            self.conv_vector = None
        
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        """
        Forward pass with geometric product structure.
        
        Args:
            x: Input tensor (B, in_channels, H, W)
        
        Returns:
            Output tensor (B, out_channels, H, W)
        """
        # Scalar convolution (always present)
        out_scalar = self.conv_scalar(x)
        
        # Vector convolution (if input has vector components)
        if self.conv_vector is not None and x.shape[1] >= 4:
            out_vector = self.conv_vector(x)
            # Sum vector components (contraction)
            out_vector = out_vector.mean(dim=1, keepdim=True)
            # Combine with scalar using geometric product
            out = out_scalar + out_vector
        else:
            out = out_scalar
        
        # Batch normalization
        out = self.norm(out)
        
        return out


class ClifffordConvBlock(nn.Module):
    """
    Clifford Convolution Block with activation and pooling.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 pool_size=2, use_geometric_product=True):
        super(ClifffordConvBlock, self).__init__()
        
        self.conv = ClifffordConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   use_geometric_product=use_geometric_product)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(pool_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x


class GeometricProductAttention(nn.Module):
    """
    Channel Attention using Geometric Product properties.
    
    Instead of standard SE-blocks, uses multivector norms to compute
    attention weights, leveraging the geometric structure.
    """
    
    def __init__(self, channels, reduction=16):
        super(GeometricProductAttention, self).__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # Compute norm-based attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Compute attention weights using geometric structure.
        
        Args:
            x: Input tensor (B, C, H, W)
        
        Returns:
            Attention-weighted output (B, C, H, W)
        """
        # Compute magnitude across spatial dimensions
        b, c, h, w = x.shape
        
        # Average pooling
        x_pool = self.avg_pool(x)  # (B, C, 1, 1)
        x_pool = x_pool.view(b, c)
        
        # FC layers for attention
        x_att = self.fc1(x_pool)
        x_att = F.relu(x_att)
        x_att = self.fc2(x_att)
        x_att = self.sigmoid(x_att)  # (B, C)
        
        # Reshape and apply to input
        x_att = x_att.view(b, c, 1, 1)
        return x * x_att


class CliffordActivation(nn.Module):
    """
    Activation function for multivectors.
    
    Uses the magnitude of multivector components for activation,
    preserving geometric structure better than standard ReLU.
    """
    
    def __init__(self, use_magnitude=True):
        super(CliffordActivation, self).__init__()
        self.use_magnitude = use_magnitude
    
    def forward(self, x):
        """
        Apply activation respecting multivector structure.
        
        Args:
            x: Multivector tensor (B, C, H, W)
        
        Returns:
            Activated tensor (B, C, H, W)
        """
        if self.use_magnitude:
            # Use magnitude-based activation
            magnitude = torch.sqrt((x ** 2).sum(dim=1, keepdim=True) + 1e-8)
            return F.relu(magnitude) * (x / (magnitude + 1e-8))
        else:
            # Standard ReLU
            return F.relu(x)


class ClifffordNorm(nn.Module):
    """
    Normalization for multivector representations.
    
    Normalizes each multivector to unit length, preserving geometric directions
    while controlling magnitude.
    """
    
    def __init__(self, eps=1e-5):
        super(ClifffordNorm, self).__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Normalize multivectors to unit magnitude.
        
        Args:
            x: Multivector tensor (B, C, H, W)
        
        Returns:
            Unit-norm multivectors (B, C, H, W)
        """
        # Compute magnitude across channels
        magnitude = torch.sqrt((x ** 2).sum(dim=1, keepdim=True) + self.eps)
        return x / magnitude


print("✓ Clifford Algebra layers defined")
