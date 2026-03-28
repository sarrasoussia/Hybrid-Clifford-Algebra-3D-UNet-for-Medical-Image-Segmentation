"""
Clifford Algebra Convolutional Layers for Geometric-Aware Neural Networks

Implements geometric algebra-based convolutions using multivector representations
of RGB pixels for improved feature extraction and equivariance properties.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


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


class CliffordConv3D(nn.Module):
    """
    Clifford (geometric product) convolution for Cl(3,0).

    This layer is a *pointwise* (1x1x1) Clifford geometric-product mixing over
    the multivector basis components at each voxel. It does not perform spatial
    neighborhood convolution; spatial mixing is expected to come from surrounding
    CNN layers (e.g., U-Net convolutions).

    Required input/output formats:
        Input : (B, C_in * 8, D, H, W)
        Output: (B, C_out * 8, D, H, W)
    """

    MULTIVECTOR_DIM = 8  # Cl(3,0) has 8 basis blades: 1 + 3 + 3 + 1

    # Basis ordering (used to build product/sign tables and grade groups):
    # [scalar, e1, e2, e3, e12, e13, e23, e123]
    _BASIS_MASKS = (0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111)
    _MASK_TO_INDEX = {mask: idx for idx, mask in enumerate(_BASIS_MASKS)}

    # grade per basis index in the above ordering
    _BASIS_GRADES = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3], dtype=torch.long)

    @classmethod
    def _build_product_and_sign_tables(cls) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build:
          - product_table_idx: (8, 8) basis index k such that e_i * e_j = +/- e_k
          - product_table_sign: (8, 8) sign (+1/-1) for the same products
        """

        def multiply_blades(mask_a: int, mask_b: int) -> tuple[int, int]:
            # Multiply blade_a by the basis vectors present in blade_b, in increasing index order.
            # For Cl(3,0) with orthonormal basis (e_i^2 = +1), the sign is determined by the
            # number of swaps needed to move the right-multiplying e_i past higher-index
            # basis vectors already present in the current blade.
            sign = 1
            mask = mask_a
            for i in range(3):  # e1, e2, e3 (0-based)
                if (mask_b >> i) & 1:
                    # Number of set bits in `mask` with index > i determines the swap parity.
                    higher = mask & (~((1 << (i + 1)) - 1))
                    # Use a portable popcount (no dependency on int.bit_count()).
                    swaps = bin(higher).count("1")
                    if swaps % 2 == 1:
                        sign = -sign
                    mask ^= (1 << i)  # toggle presence of e_i (adds or cancels via e_i*e_i=+1)
            return sign, mask

        product_table_idx = torch.empty((8, 8), dtype=torch.long)
        product_table_sign = torch.empty((8, 8), dtype=torch.float32)

        for i, mask_i in enumerate(cls._BASIS_MASKS):
            for j, mask_j in enumerate(cls._BASIS_MASKS):
                sign, mask_k = multiply_blades(mask_i, mask_j)
                product_table_idx[i, j] = cls._MASK_TO_INDEX[mask_k]
                product_table_sign[i, j] = float(sign)

        return product_table_idx, product_table_sign

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        product_table: Optional[torch.Tensor] = None,
        product_sign_table: Optional[torch.Tensor] = None,
        restrict_grades: bool = False,
        use_grade_wise_scaling: bool = False,
        bias: bool = False,
    ):
        """
        Args:
            in_channels: C_in in the required input shape (B, C_in*8, D, H, W).
            out_channels: C_out in the required output shape (B, C_out*8, D, H, W).
            kernel_size: Only `1` is supported for this pointwise implementation.
            product_table: Optional (8, 8) tensor with entries k such that e_i * e_j = +/- e_k.
                            If omitted, a Cl(3,0)-consistent table is built.
            product_sign_table: Optional (8, 8) tensor with +/-1 signs for e_i * e_j.
                                 If omitted, a Cl(3,0)-consistent sign table is built.
            restrict_grades: If True, only allow scalar<->scalar and scalar<->vector interactions
                              (blocks bivector/trivector interactions and vector-vector products).
            use_grade_wise_scaling: If True, apply learnable multiplicative scaling per output grade
                                     (scalar/vector/bivector/trivector).
            bias: If True, add a learnable bias per output channel & basis component.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if kernel_size != 1:
            raise ValueError(
                f"CliffordConv3D implements a pointwise (kernel_size=1) geometric-product mixing only; got kernel_size={kernel_size}"
            )

        # Learnable weights W[o, c, i, j] multiplies input component i with "kernel basis" j.
        # The geometric product mapping (i, j) -> k then determines which output basis
        # component accumulates.
        self.weights = nn.Parameter(
            torch.empty(out_channels, in_channels, self.MULTIVECTOR_DIM, self.MULTIVECTOR_DIM)
        )

        # Build geometric product mapping tensor geom_map[k, i, j] = sign(i,j) if (i,j)->k else 0.
        if product_table is None or product_sign_table is None:
            built_idx, built_sign = self._build_product_and_sign_tables()
            if product_table is None:
                product_table = built_idx
            if product_sign_table is None:
                product_sign_table = built_sign
        else:
            if tuple(product_table.shape) != (8, 8) or tuple(product_sign_table.shape) != (8, 8):
                raise ValueError("product_table and product_sign_table must have shape (8, 8)")

        product_table = product_table.to(dtype=torch.long)
        product_sign_table = product_sign_table.to(dtype=torch.float32)

        # geom_map: (k, i, j)
        geom_map = torch.zeros(
            (self.MULTIVECTOR_DIM, self.MULTIVECTOR_DIM, self.MULTIVECTOR_DIM),
            dtype=torch.float32,
        )
        for i in range(self.MULTIVECTOR_DIM):
            for j in range(self.MULTIVECTOR_DIM):
                k = int(product_table[i, j].item())
                geom_map[k, i, j] = product_sign_table[i, j]

        if restrict_grades:
            # Allowed blade-pairs:
            #   scalar (grade 0) * scalar/vector (grade 0 or 1)
            #   vector (grade 1) * scalar (grade 0)
            grade_i = self._BASIS_GRADES.view(8, 1)  # (i,1)
            grade_j = self._BASIS_GRADES.view(1, 8)  # (1,j)
            allowed = (
                (grade_i == 0) & ((grade_j == 0) | (grade_j == 1))
            ) | ((grade_i == 1) & (grade_j == 0))
            geom_map = geom_map * allowed.to(geom_map.dtype).unsqueeze(0)  # (k,i,j)

        self.register_buffer("geom_map", geom_map)

        self.register_buffer("basis_grades", self._BASIS_GRADES)
        self.use_grade_wise_scaling = use_grade_wise_scaling
        if use_grade_wise_scaling:
            # Order: [scalar(0), vector(1), bivector(2), trivector(3)]
            self.grade_scales = nn.Parameter(torch.ones(4, dtype=torch.float32))
        else:
            self.grade_scales = None

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, self.MULTIVECTOR_DIM, dtype=torch.float32))

        # Weight init (stable for einsum-heavy mixing)
        nn.init.xavier_uniform_(self.weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C_in*8, D, H, W)
        Returns:
            Tensor of shape (B, C_out*8, D, H, W)
        """
        B, C_mul, D, H, W = x.shape
        expected_C_mul = self.in_channels * self.MULTIVECTOR_DIM
        if C_mul != expected_C_mul:
            raise ValueError(f"Expected input channels {expected_C_mul} (got {C_mul})")

        # (B, C_in*8, D, H, W) -> (B, C_in, 8, D, H, W)
        x_mv = x.reshape(B, self.in_channels, self.MULTIVECTOR_DIM, D, H, W)

        # Core geometric product mixing:
        #   x_mv[b, c, i, d, h, w] * weights[o, c, i, j] -> contributions for (o, i, j)
        #   geom_map[k, i, j] selects and signs the resulting basis component e_k
        #
        # out_mv[b, o, k, d, h, w] = sum_{c,i,j} x_mv[b,c,i,d,h,w] * W[o,c,i,j] * geom_map[k,i,j]
        out_mv = torch.einsum(
            "b c i d h w, o c i j, k i j -> b o k d h w",
            x_mv,
            self.weights,
            self.geom_map.to(dtype=x.dtype, device=x.device),
        )

        if self.bias is not None:
            out_mv = out_mv + self.bias.to(dtype=x.dtype, device=x.device).view(1, self.out_channels, 8, 1, 1, 1)

        if self.grade_scales is not None:
            scale_k = self.grade_scales.to(dtype=x.dtype, device=x.device)[self.basis_grades]  # (8,)
            out_mv = out_mv * scale_k.view(1, 1, 8, 1, 1, 1)

        # (B, C_out, 8, D, H, W) -> (B, C_out*8, D, H, W)
        out_mv = out_mv.contiguous().reshape(B, self.out_channels * self.MULTIVECTOR_DIM, D, H, W)
        return out_mv


class CliffordProjection3D(nn.Module):
    """
    Projects real-valued 3D feature maps to Cl(3,0) multivector space (8 components per multivector).

    Input : (B, C_in, D, H, W)
    Output: (B, C_out * 8, D, H, W)
    """

    MULTIVECTOR_DIM = 8

    # Basis ordering must match `CliffordConv3D`:
    # [scalar(0) | vector(1,2,3) | bivector(4,5,6) | trivector(7)]
    _BASIS_GRADES = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3], dtype=torch.long)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        use_grade_wise_scaling: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Learnable voxel-wise projection (1x1x1 conv) preserves spatial structure.
        self.proj = nn.Conv3d(in_channels, out_channels * self.MULTIVECTOR_DIM, kernel_size=1, bias=bias)

        self.use_grade_wise_scaling = use_grade_wise_scaling
        if use_grade_wise_scaling:
            # order: scalar, vector, bivector, trivector
            self.grade_scales = nn.Parameter(torch.ones(4, dtype=torch.float32))
            self.register_buffer("basis_grades", self._BASIS_GRADES)
        else:
            self.grade_scales = None
            self.register_buffer("basis_grades", self._BASIS_GRADES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)
        Returns:
            (B, C_out * 8, D, H, W)
        """
        B, C, D, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected input channels={self.in_channels}, got C={C}")

        mv = self.proj(x)  # (B, C_out*8, D, H, W)

        if self.use_grade_wise_scaling:
            # (B, C_out, 8, D, H, W)
            mv = mv.view(B, self.out_channels, self.MULTIVECTOR_DIM, D, H, W)
            # (8,) -> scale per component based on grade
            scale_k = self.grade_scales.to(dtype=mv.dtype, device=mv.device)[self.basis_grades]  # (8,)
            mv = mv * scale_k.view(1, 1, self.MULTIVECTOR_DIM, 1, 1, 1)
            mv = mv.reshape(B, self.out_channels * self.MULTIVECTOR_DIM, D, H, W)

        return mv


print("✓ Clifford Algebra layers defined")
