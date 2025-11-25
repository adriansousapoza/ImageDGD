import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Any
from torch import Tensor

try:
    from .pca import TorchPCA
except ImportError:
    from pca import TorchPCA

class RepresentationLayer(nn.Module):
    """
    Representation layer storing learned embeddings optimized during training.
    
    Supports multiple multivariate initialization distributions and provides utilities for
    representation manipulation and analysis. All distributions are multivariate versions
    that operate in d-dimensional space where d is the representation dimensionality.
    """
    
    AVAILABLE_DISTS = ["normal", "uniform", "laplace", "student_t", "uniform_ball", "uniform_sphere", "logistic", "hyperbolic", "zeros", "pca"]
    
    def __init__(self, 
                 dim: int,
                 n_samples: Optional[int] = None,
                 values: Optional[Tensor] = None, 
                 dist: str = "normal", 
                 dist_params: Optional[Dict[str, Union[int, float]]] = None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        """Initialize representation layer.

        Parameters
        ----------
        dim : int
            Dimensionality of representations.
        n_samples : int, optional
            Number of samples. If values provided, derived from values unless 
            explicitly specified (must match if both provided).
        values : torch.Tensor, optional
            Reference tensor for shape derivation. Representations still sampled 
            from specified distribution.
        dist : str, default "normal"
            Distribution for sampling: "normal", "uniform", "laplace", "student_t", "uniform_ball", "uniform_sphere", "logistic", "hyperbolic", "zeros", "pca".
        dist_params : dict, optional
            Distribution-specific parameters for multivariate distributions:
            - normal: mean (d-dim vector or scalar, default: zeros), cov (d×d matrix or scalar, default: identity)
            - uniform: low (scalar, default: -1.0), high (scalar, default: 1.0)  
            - laplace: loc (d-dim vector or scalar, default: zeros), scale_matrix (d×d matrix or scalar, default: identity), rate (scalar, default: 1.0)
            - student_t: df (scalar, default: 3.0), scale_matrix (d×d matrix or scalar, default: identity)
            - uniform_ball: radius (scalar, default: 1.0)
            - uniform_sphere: radius (scalar, default: 1.0)
            - logistic: loc (d-dim vector or scalar, default: zeros), scale (d-dim vector or scalar, default: ones)
            - hyperbolic: mu (d-dim vector or scalar, default: zeros), alpha (scalar, default: 1.5), beta (scalar, default: 0.0), delta (scalar, default: 1.0)
            - pca: data (required tensor), whiten (bool, default: False), svd_solver (str, default: "auto")
            - zeros: none
            
            Note: When scalars are provided for vector/matrix parameters:
            - Scalar mean/loc → broadcast to d-dimensional vector
            - Scalar cov/scale_matrix → spherical covariance/scale (σ²I_d)
        device : str or torch.device, optional
            Device for representations.
        """
        # Setup
        self.device = device or (values.device if values is not None else 
                               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        super().__init__()
        
        if dist_params is None:
            dist_params = {}
        
        # Validate required parameters
        if n_samples is None and values is None:
            raise ValueError("Either 'n_samples' or 'values' must be provided")
        
        # Handle values tensor
        if values is not None:
            values_n_samples, values_dim = values.shape[0], values.shape[-1]
            
            if n_samples is not None and n_samples != values_n_samples:
                raise ValueError(f"Mismatch between n_samples ({n_samples}) and values shape ({values_n_samples})")
            if dim != values_dim:
                raise ValueError(f"Mismatch between dim ({dim}) and values shape ({values_dim})")
            
            n_samples = n_samples or values_n_samples
        
        # Set dimensions before distribution sampling
        self._n_rep = n_samples
        self._dim = dim
        
        # Initialize based on distribution
        dist_methods = {
            "normal": self._get_rep_from_normal,
            "uniform": self._get_rep_from_uniform,
            "laplace": self._get_rep_from_laplace,
            "student_t": self._get_rep_from_student_t,
            "uniform_ball": self._get_rep_from_uniform_ball,
            "uniform_sphere": self._get_rep_from_uniform_sphere,
            "zeros": self._get_rep_from_zeros,
            "logistic": self._get_rep_from_logistic,
            "hyperbolic": self._get_rep_from_hyperbolic,
            "pca": self._get_rep_from_pca
        }
        
        if dist not in dist_methods:
            available = ", ".join(f'"{d}"' for d in self.AVAILABLE_DISTS)
            raise ValueError(f"Unsupported distribution '{dist}'. Available: {available}")
        
        self._z, self._options = dist_methods[dist](dist_params)
        self._z = self._z.to(self.device)


    def _get_rep_from_zeros(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Initialize representations as zeros.
        
        Mathematical formulation:
            X = 0 ∈ ℝᵈ
            
        Where d is the dimensionality of the representation space.
        """
        z = nn.Parameter(torch.zeros(self._n_rep, self._dim, device=self.device), requires_grad=True)
        return z, {"dist_name": "zeros"}
    
    def _get_rep_from_pca(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Initialize representations from PCA projection of input data.
        
        Mathematical formulation:
            Given data matrix X ∈ ℝⁿˣᵖ, compute PCA projection:
            Z = (X - μ) * V_k
            
        Where:
            - μ is the mean of X
            - V_k contains the first k principal components
            - k = dim (number of components)
            
        Parameters in options:
            - data: torch.Tensor (required), shape (n_samples, n_features)
            - whiten: bool (default: False), whether to whiten components
            - svd_solver: str (default: "auto"), SVD solver method
        """
        if 'data' not in options:
            raise ValueError("PCA initialization requires 'data' parameter in dist_params")
        
        data = options['data']
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be a torch.Tensor")
        
        # Reshape to 2D if needed
        if len(data.shape) > 2:
            n_samples = data.shape[0]
            data = data.reshape(n_samples, -1)
        
        if data.shape[0] != self._n_rep:
            raise ValueError(f"Number of data samples ({data.shape[0]}) must match n_samples ({self._n_rep})")
        
        # Move data to device
        data = data.to(self.device)
        
        # Get PCA parameters
        whiten = options.get('whiten', False)
        svd_solver = options.get('svd_solver', 'auto')
        
        # Apply PCA
        pca = TorchPCA(n_components=self._dim, whiten=whiten, svd_solver=svd_solver)
        transformed_data = pca.fit_transform(data)
        
        # Create parameter from transformed data
        z = nn.Parameter(transformed_data.clone(), requires_grad=True)
        
        return z, {
            "dist_name": "pca",
            "whiten": whiten,
            "svd_solver": svd_solver,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist() if pca.explained_variance_ratio_ is not None else None
        }
    
    def _get_rep_from_uniform(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations from multivariate uniform distribution.
        
        Mathematical formulation:
            X ~ U([a₁,b₁] × [a₂,b₂] × ... × [aᵈ,bᵈ])
            
        For simplicity, we use the same bounds for all dimensions:
            Xᵢ ~ U(low, high) independently for i = 1,...,d
            
        Where:
            - low, high: bounds of the uniform distribution
            - d: dimensionality of the representation space
        """
        low, high = options.get("low", -1.0), options.get("high", 1.0)
        
        z = nn.Parameter(
            torch.empty(self._n_rep, self._dim, device=self.device).uniform_(low, high),
            requires_grad=True
        )
        return z, {"dist_name": "uniform", "low": low, "high": high}
    
    def _get_rep_from_uniform_ball(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations uniformly from a d-dimensional ball.
        
        Mathematical formulation:
            X ~ Uniform(B_d(0, r))
            
        Where B_d(0, r) = {x ∈ ℝᵈ : ||x||₂ ≤ r} is the d-dimensional ball
        of radius r centered at the origin.
        
        Sampling method:
            1. Generate U ~ Uniform(S^(d-1)) (uniform on unit sphere)
            2. Generate R ~ U(0,1)^(1/d) (radius with correct volume scaling)
            3. X = r × R × U
            
        Where:
            - r: radius of the ball
            - d: dimensionality of the representation space
        """
        radius = options.get("radius", 1.0)

        # Generate random directions and scale by random radii
        normal_samples = torch.randn(self._n_rep, self._dim, device=self.device)
        unit_directions = normal_samples / torch.norm(normal_samples, dim=1, keepdim=True)
        random_radii = radius * torch.rand(self._n_rep, 1, device=self.device).pow(1.0 / self._dim)
        
        z = nn.Parameter(unit_directions * random_radii, requires_grad=True)
        return z, {"dist_name": "uniform_ball", "radius": radius}
    
    def _get_rep_from_uniform_sphere(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations uniformly from a d-dimensional sphere surface.
        
        Mathematical formulation:
            X ~ Uniform(S^(d-1)(0, r))
            
        Where S^(d-1)(0, r) = {x ∈ ℝᵈ : ||x||₂ = r} is the (d-1)-dimensional sphere
        surface of radius r centered at the origin.
        
        Sampling method:
            1. Generate Z ~ N(0, I_d) (multivariate standard normal)
            2. Normalize: U = Z / ||Z||₂ (uniform on unit sphere surface)
            3. Scale: X = r × U
            
        This method is based on the mathematical fact that if Z ~ N(0, I), then
        Z/||Z|| is uniformly distributed on the unit sphere surface. This works
        because the multivariate normal distribution is rotationally invariant,
        and there is only one rotationally invariant distribution on the sphere surface:
        the uniform distribution.
        
        Where:
            - r: radius of the sphere surface
            - d: dimensionality of the representation space
        """
        radius = options.get("radius", 1.0)

        # Generate samples from multivariate standard normal
        normal_samples = torch.randn(self._n_rep, self._dim, device=self.device)
        
        # Normalize to unit sphere surface (no radius scaling like in uniform_ball)
        norms = torch.norm(normal_samples, dim=1, keepdim=True)
        # Avoid division by zero (extremely unlikely but mathematically sound)
        norms = torch.clamp(norms, min=1e-8)
        unit_directions = normal_samples / norms
        
        # Scale by desired radius
        z = nn.Parameter(radius * unit_directions, requires_grad=True)
        return z, {"dist_name": "uniform_sphere", "radius": radius}

    def _get_rep_from_normal(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations from multivariate normal distribution.
        
        Mathematical formulation:
            X ~ 𝒩(μ, Σ)
            
        Where:
            - μ ∈ ℝᵈ: mean vector (scalar will be broadcast to vector)
            - Σ ∈ ℝᵈˣᵈ: covariance matrix (scalar will create spherical covariance σ²I)
            - d: dimensionality of the representation space
            
        Probability density function:
            f(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
        """
        mean = options.get("mean", torch.zeros(self._dim))
        cov = options.get("cov", torch.eye(self._dim))
        
        # Handle scalar mean (broadcast to vector)
        if isinstance(mean, (int, float)):
            mean = torch.full((self._dim,), mean)
        
        # Handle scalar covariance (create spherical covariance matrix)
        if isinstance(cov, (int, float)):
            cov = cov * torch.eye(self._dim)
        
        mvn = torch.distributions.MultivariateNormal(mean, cov)
        samples = mvn.sample((self._n_rep,)).to(self.device)
        
        z = nn.Parameter(samples, requires_grad=True)
        return z, {"dist_name": "normal", "mean": mean, "cov": cov}
    
    def _get_rep_from_student_t(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations from multivariate Student's t distribution.
        
        Mathematical formulation:
            X ~ t_ν(0, Σ)
            
        Where:
            - ν > 0: degrees of freedom
            - Σ ∈ ℝᵈˣᵈ: scale matrix (scalar will create spherical scale σ²I)
            - d: dimensionality of the representation space
            
        Probability density function:
            f(x) = Γ((ν+d)/2) / [Γ(ν/2)(νπ)^(d/2)|Σ|^(1/2)] × (1 + x^T Σ^(-1) x / ν)^(-(ν+d)/2)
            
        Sampling method (using normal-gamma mixture):
            1. Z ~ 𝒩(0, Σ)
            2. W ~ Gamma(ν/2, ν/2)
            3. X = Z / √(W/ν)
        """
        df = options.get("df", 3.0)
        scale_matrix = options.get("scale_matrix", torch.eye(self._dim))
        
        # Handle scalar scale_matrix (create spherical scale matrix)
        if isinstance(scale_matrix, (int, float)):
            scale_matrix = scale_matrix * torch.eye(self._dim)
        
        # Multivariate t = (X - μ) / sqrt(W/ν) where X ~ MVN, W ~ Gamma
        mvn = torch.distributions.MultivariateNormal(torch.zeros(self._dim), scale_matrix)
        gamma = torch.distributions.Gamma(df/2, df/2)
        
        normal_samples = mvn.sample((self._n_rep,)).to(self.device)
        gamma_samples = gamma.sample((self._n_rep,)).to(self.device).unsqueeze(-1)
        
        samples = normal_samples / torch.sqrt(gamma_samples / df)
        z = nn.Parameter(samples, requires_grad=True)
        return z, {"dist_name": "mvt", "df": df, "scale_matrix": scale_matrix}



    def _get_rep_from_laplace(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations from multivariate Laplace distribution using scale mixture representation.
        
        Mathematical formulation:
            X ~ ℒ(μ, Σ)
            
        Where:
            - μ ∈ ℝᵈ: location vector (scalar will be broadcast to vector)
            - Σ ∈ ℝᵈˣᵈ: scale matrix (scalar will create spherical scale σ²I)
            - d: dimensionality of the representation space
            
        Probability density function:
            f(x) = (2π)^(-d/2) |Σ|^(-1/2) ∫₀^∞ w^(-d/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ)/w - λw) dw
            
        Sampling method (using scale mixture of normals):
            1. W ~ Exponential(λ) (mixing weights)
            2. Z ~ 𝒩(0, Σ)
            3. X = μ + √W × Z
            
        This representation gives the multivariate Laplace distribution
        with location μ and scale matrix Σ.
        """
        # Parameters with defaults
        loc = options.get("loc", torch.zeros(self._dim, device=self.device))
        scale_matrix = options.get("scale_matrix", torch.eye(self._dim, device=self.device))
        rate = options.get("rate", 1.0)  # Rate parameter for exponential distribution
        
        # Handle scalar location (broadcast to vector)
        if isinstance(loc, (int, float)):
            loc = torch.full((self._dim,), loc, device=self.device)
        else:
            loc = loc.to(self.device)
            
        # Handle scalar scale_matrix (create spherical scale matrix)
        if isinstance(scale_matrix, (int, float)):
            scale_matrix = scale_matrix * torch.eye(self._dim, device=self.device)
        else:
            scale_matrix = scale_matrix.to(self.device)
        
        # Sample from multivariate normal
        mvn = torch.distributions.MultivariateNormal(torch.zeros(self._dim, device=self.device), scale_matrix)
        normal_samples = mvn.sample((self._n_rep,)).to(self.device)
        
        # Sample mixing weights from exponential distribution  
        exp_dist = torch.distributions.Exponential(rate)
        mixing_weights = exp_dist.sample((self._n_rep,)).to(self.device).unsqueeze(-1)  # Shape: (n_rep, 1)
        
        # Scale mixture: X = μ + sqrt(W) * Z
        samples = loc.unsqueeze(0) + torch.sqrt(mixing_weights) * normal_samples
        
        z = nn.Parameter(samples, requires_grad=True)
        return z, {
            "dist_name": "mvlaplace", 
            "loc": loc, 
            "scale_matrix": scale_matrix, 
            "rate": rate
        }


    def _get_rep_from_logistic(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations from multivariate logistic distribution.
        
        Mathematical formulation:
            For each dimension i: X_i ~ Logistic(μᵢ, sᵢ)
            
        Where:
            - μ ∈ ℝᵈ: location vector (scalar will be broadcast to vector)
            - s ∈ ℝᵈ₊: scale vector (scalar will be broadcast to vector)
            - d: dimensionality of the representation space
            
        Probability density function (univariate):
            f(x; μ, s) = e^(-(x-μ)/s) / (s(1 + e^(-(x-μ)/s))²)
            
        Sampling method using quantile function:
            1. U ~ Uniform(0, 1)
            2. X = μ + s × log(U/(1-U))  (logit function)
        """
        # Parameters with defaults
        loc = options.get("loc", torch.zeros(self._dim, device=self.device))
        scale = options.get("scale", torch.ones(self._dim, device=self.device))
        
        # Handle scalar location (broadcast to vector)
        if isinstance(loc, (int, float)):
            loc = torch.full((self._dim,), loc, device=self.device)
        else:
            loc = loc.to(self.device)
            
        # Handle scalar scale (broadcast to vector)
        if isinstance(scale, (int, float)):
            scale = torch.full((self._dim,), scale, device=self.device)
        else:
            scale = scale.to(self.device)
        
        # Ensure positive scale parameters
        if torch.any(scale <= 0):
            raise ValueError("Scale parameters must be positive")
        scale = torch.abs(scale)  # Additional safety
        
        # Sample from uniform distribution
        uniform_samples = torch.rand(self._n_rep, self._dim, device=self.device)
        
        # Apply inverse CDF (quantile function): Q(p) = μ + s * log(p/(1-p))
        # Add small epsilon to avoid numerical issues at boundaries
        eps = 1e-7
        uniform_samples = torch.clamp(uniform_samples, eps, 1 - eps)
        
        logit_samples = torch.log(uniform_samples / (1 - uniform_samples))
        samples = loc.unsqueeze(0) + scale.unsqueeze(0) * logit_samples
        
        z = nn.Parameter(samples, requires_grad=True)
        return z, {
            "dist_name": "logistic",
            "loc": loc,
            "scale": scale
        }


    def _get_rep_from_hyperbolic(self, options: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        """Sample representations from multivariate hyperbolic distribution using normal-variance mixture.
        
        Mathematical formulation:
            X ~ Hyperbolic(μ, α, β, δ)
            
        Where:
            - μ ∈ ℝᵈ: location vector (scalar will be broadcast to vector)
            - α > |β|: tail heaviness parameter
            - β ∈ (-α, α): asymmetry parameter  
            - δ > 0: scale parameter
            - γ = √(α² - β²): derived parameter
            
        Sampling method using normal-variance mixture:
            1. W ~ GeneralizedInverseGaussian(λ=1, δγ, γ)
            2. Z ~ N(0, I)
            3. X = μ + βδ²W/γ + δ√W × Z
            
        This gives a multivariate hyperbolic distribution where each component
        follows the same marginal hyperbolic distribution.
        """
        # Parameters with defaults
        mu = options.get("mu", torch.zeros(self._dim, device=self.device))
        alpha = options.get("alpha", 1.5)
        beta = options.get("beta", 0.0)  
        delta = options.get("delta", 1.0)
        
        # Handle scalar location (broadcast to vector)
        if isinstance(mu, (int, float)):
            mu = torch.full((self._dim,), mu, device=self.device)
        else:
            mu = mu.to(self.device)
        
        # Validate parameters
        if not (alpha > abs(beta)):
            raise ValueError(f"Must have α > |β|, got α={alpha}, β={beta}")
        if delta <= 0:
            raise ValueError(f"δ must be positive, got δ={delta}")
        
        # Warn about asymmetry when beta != 0
        if beta != 0.0:
            import warnings
            warnings.warn(f"β={beta} ≠ 0: Distribution will be asymmetric. "
                         f"Use β=0 for symmetric hyperbolic distribution.", 
                         UserWarning)
            
        gamma = torch.sqrt(torch.tensor(alpha**2 - beta**2, device=self.device))
        
        # Approximate GIG sampling using Gamma approximation for simplicity
        # For λ=1, GIG(1, a, b) ≈ Gamma(shape, rate) where we match first two moments
        # This is a reasonable approximation - exact GIG sampling would require more complex methods
        # like acceptance-rejection or specialized algorithms
        
        gig_mean = delta / gamma  # Approximate mean of GIG(1, δγ, γ)
        gig_var = delta / (gamma**3)  # Approximate variance
        
        # Match Gamma distribution moments: mean = shape/rate, var = shape/rate²
        rate = gig_mean / gig_var
        shape = gig_mean * rate
        
        # Sample mixing variables W ~ Gamma (approximating GIG)
        gamma_dist = torch.distributions.Gamma(shape, rate)
        W = gamma_dist.sample((self._n_rep,)).to(self.device)
        
        # Sample from standard multivariate normal
        Z = torch.randn(self._n_rep, self._dim, device=self.device)
        
        # Construct hyperbolic samples: X = μ + βδ²W/γ + δ√W × Z
        drift_term = beta * delta**2 * W.unsqueeze(-1) / gamma
        diffusion_term = delta * torch.sqrt(W).unsqueeze(-1) * Z
        
        samples = mu.unsqueeze(0) + drift_term + diffusion_term
        
        z = nn.Parameter(samples, requires_grad=True)
        return z, {
            "dist_name": "hyperbolic",
            "mu": mu,
            "alpha": alpha,
            "beta": beta, 
            "delta": delta,
            "gamma": gamma.item()
        }



    @property
    def n_rep(self) -> int:
        """Number of representations."""
        return self._n_rep

    @n_rep.setter
    def n_rep(self, value: int) -> None:
        raise ValueError("n_rep is read-only. Create new instance to change.")

    @property
    def dim(self) -> int:
        """Dimensionality of representations."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        raise ValueError("dim is read-only. Create new instance to change.")

    @property
    def options(self) -> Dict[str, Any]:
        """Distribution options used for initialization."""
        return self._options

    @options.setter
    def options(self, value: Dict[str, Any]) -> None:
        raise ValueError("options is read-only. Create new instance to change.")
    
    @property
    def z(self) -> Tensor:
        """Representation values."""
        return self._z

    @z.setter
    def z(self, value: Tensor) -> None:
        raise ValueError("z is read-only. Create new instance to change.")

    def forward(self, ixs: Optional[Union[List[int], Tensor]] = None, 
                index_map: Optional[Dict[int, int]] = None,
                batch_size: Optional[int] = None) -> Tensor:
        """Forward pass returning representation values.

        Parameters
        ----------
        ixs : list or Tensor, optional
            Sample indices to return. If None, returns all representations.
        index_map : dict, optional
            Mapping from dataset indices to representation indices.
        batch_size : int, optional
            Process in batches for memory efficiency.

        Returns
        -------
        torch.Tensor
            Requested representations.
        """
        if ixs is None:
            return self.z
        
        # Handle index mapping
        if index_map is not None:
            if isinstance(ixs, torch.Tensor):
                mapped_ixs = torch.tensor([index_map.get(idx.item(), 0) for idx in ixs], device=ixs.device)
            else:
                mapped_ixs = [index_map.get(idx, 0) for idx in ixs]
        else:
            # Clamp indices to valid range
            if isinstance(ixs, torch.Tensor):
                mapped_ixs = torch.clamp(ixs, 0, len(self.z)-1)
            else:
                mapped_ixs = [min(max(0, idx), len(self.z)-1) for idx in ixs]
        
        # Process in batches if needed
        if batch_size is not None and len(mapped_ixs) > batch_size:
            result_chunks = []
            for i in range(0, len(mapped_ixs), batch_size):
                batch_ixs = mapped_ixs[i:i+batch_size]
                result_chunks.append(self.z[batch_ixs])
            return torch.cat(result_chunks, dim=0)
        
        return self.z[mapped_ixs]

    def to(self, device: Union[str, torch.device]) -> 'RepresentationLayer':
        """Move representations to specified device."""
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._z = self._z.to(device)
        return super().to(device)
    
    def save(self, path: str) -> None:
        """Save representations to file."""
        state_dict = {
            'z': self.z.detach().cpu(),
            'n_rep': self._n_rep,
            'dim': self._dim,
            'options': self._options
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[Union[str, torch.device]] = None) -> 'RepresentationLayer':
        """Load representations from file."""
        state_dict = torch.load(path, map_location='cpu')
        
        dim = state_dict['dim']
        n_samples = state_dict['n_rep']
        
        # Extract distribution parameters
        dist_params = {}
        if 'options' in state_dict:
            saved_options = state_dict['options']
            for key, value in saved_options.items():
                if key not in ['dist_name', 'n_samples', 'dim']:
                    dist_params[key] = value
            dist_name = saved_options.get('dist_name', 'normal')
        else:
            dist_name = 'normal'
        
        return cls(dim=dim, n_samples=n_samples, values=state_dict['z'], 
                  dist=dist_name, dist_params=dist_params, device=device)
    
    @classmethod
    def initialize_from_pca(cls, data: Tensor, n_components: Optional[int] = None, 
                            device: Optional[Union[str, torch.device]] = None) -> 'RepresentationLayer':
        """Initialize representations using PCA of input data."""
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be a torch.Tensor")
        
        # Reshape to 2D if needed
        if len(data.shape) > 2:
            n_samples = data.shape[0]
            data = data.reshape(n_samples, -1)
        
        device = device or data.device
        n_components = n_components or min(data.shape)
        
        # Apply PCA
        pca = TorchPCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        
        return cls(dim=transformed_data.shape[1], values=transformed_data, device=device)
