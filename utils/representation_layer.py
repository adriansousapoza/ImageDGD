import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Union, List, Tuple, Any
from torch import Tensor
from utils.pca import TorchPCA

class RepresentationLayer(nn.Module):
    """
    Class implementing a representation layer accumulating gradients.
    
    This layer stores learned embeddings for data samples that can be optimized
    during training. It supports various initialization methods and utility functions
    for representation manipulation and analysis.
    """

    ######################## PUBLIC ATTRIBUTE #########################

    # Set the available distributions to sample the representations from
    AVAILABLE_DISTS = ["normal", "uniform", "laplace", "student_t", "cauchy"]

    ######################### INITIALIZATION ##########################
    
    def __init__(self, 
                 values: Optional[Tensor] = None, 
                 dist: str = "normal", 
                 dist_options: Optional[Dict[str, Union[int, float]]] = None,
                 device: Optional[Union[str, torch.device]] = None) -> None:
        """Initialize a representation layer.

        Parameters
        ----------
        values : ``torch.Tensor``, optional
            A tensor used to initialize the representations in
            the layer.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations in the tensor.

            * The second dimension has a length equal to the
              dimensionality of the representations.

            If the tensor is not passed, the representations will be
            initialized by sampling the distribution specified
            by ``dist``.

        dist : ``str``, {``"normal"``, ``"uniform"``, ``"laplace"``, ``"student_t"``, ``"cauchy"``}, default: ``"normal"``
            The name of the distribution used to sample the
            representations, if no ``values`` are passed.

            By default, the distribution is a ``"normal"``
            distribution.

        dist_options : ``dict``, optional
            A dictionary containing the parameters to sample the
            representations from the distribution, if no ``values``
            are passed.

            For any distribution the following keys and associated
            parameters must be provided:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Distribution-specific parameters:
            - For ``"normal"``: 
              * ``"mean"`` : the mean (default: 0.0)
              * ``"stddev"`` : the standard deviation (default: 1.0)
            
            - For ``"uniform"``:
              * ``"low"`` : lower bound (default: -1.0)
              * ``"high"`` : upper bound (default: 1.0)
            
            - For ``"laplace"``:
              * ``"loc"`` : location parameter (default: 0.0)
              * ``"scale"`` : scale parameter (default: 1.0)
            
            - For ``"student_t"``:
              * ``"df"`` : degrees of freedom (default: 3.0)
              * ``"scale"`` : scale parameter (default: 1.0)
            
            - For ``"cauchy"``:
              * ``"scale"`` : scale parameter (default: 1.0)
        """
        # Setup device
        self.device = device or (values.device if values is not None else 
                               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize an instance of the 'nn.Module' class.
        super().__init__()
        
        # If a tensor of values was passed
        if values is not None:
            # Set the options used to initialize the representations
            # to an empty dictionary, since they have not been 
            # sampled from any distribution.
            self._options = {}

            # Get the number of representations, the
            # dimensionality of the representations, and the values
            # of the representations from the tensor.
            self._n_rep, self._dim, self._z = self._get_rep_from_values(values=values)
        
        # Otherwise
        else:
            # Choose initialization method based on distribution
            if dist == "normal":
                self._n_rep, self._dim, self._z, self._options = self._get_rep_from_normal(options=dist_options)
            elif dist == "uniform":
                self._n_rep, self._dim, self._z, self._options = self._get_rep_from_uniform(options=dist_options)
            elif dist == "laplace":
                self._n_rep, self._dim, self._z, self._options = self._get_rep_from_laplace(options=dist_options)
            elif dist == "student_t":
                self._n_rep, self._dim, self._z, self._options = self._get_rep_from_student_t(options=dist_options)
            elif dist == "cauchy":
                self._n_rep, self._dim, self._z, self._options = self._get_rep_from_cauchy(options=dist_options)
            else:
                # Raise an error for unsupported distribution
                available_dists_str = ", ".join(f'"{d}"' for d in self.AVAILABLE_DISTS)
                errstr = (f"Unsupported distribution '{dist}'. The only distributions from which "
                         f"it is possible to sample the representations are: {available_dists_str}.")
                raise ValueError(errstr)
        
        # Move to the specified device
        self._z = self._z.to(self.device)

    def _get_rep_from_values(self, values: Tensor) -> Tuple[int, int, Tensor]:
        """Get the representations from a given tensor of values.

        Parameters
        ----------
        values : ``torch.Tensor``
            The tensor used to initialize the representations.

        Returns
        -------
        n_rep : ``int``
            The number of representations found in the input tensor.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """
        # Get the number of representations from the first dimension of
        # the tensor.
        n_rep = values.shape[0]
        
        # Get the dimensionality of the representations from the last
        # dimension of the tensor.
        dim = values.shape[-1]

        # Initialize a tensor with the representations.
        z = nn.Parameter(torch.zeros_like(values), requires_grad=True)

        # Fill the tensor with the given values.
        with torch.no_grad():
            z.copy_(values)

        # Return the number of representations, the dimensionality of
        # the representations, and the values of the representations.
        return n_rep, dim, z

    def _get_rep_from_normal(self, options: Dict[str, Any]) -> Tuple[int, int, Tensor, Dict[str, Any]]:
        """Get the representations by sampling from a normal distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a normal distribution.

            The dictionary must contains the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the normal distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the normal distribution.

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations (default: 0.0).

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations (default: 1.0).

        Returns
        -------
        n_rep : ``int``
            The number of representations found in the input tensor.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """
        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the mean of the normal distribution from which the
        # representations should be samples.
        mean = options.get("mean", 0.0)

        # Get the standard deviation of the normal distribution
        # from which the representations should be sampled.
        stddev = options.get("stddev", 1.0)

        # Get the values of the representations.
        z = nn.Parameter(
            torch.normal(mean, stddev, size=(n_rep, dim), device=self.device),
            requires_grad=True
        )
        
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
        return n_rep, dim, z, {"dist_name": "normal", "mean": mean, "stddev": stddev}

    def _get_rep_from_uniform(self, options: Dict[str, Any]) -> Tuple[int, int, Tensor, Dict[str, Any]]:
        """Get the representations by sampling from a uniform distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a uniform distribution.

            The dictionary must contains the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            Optional parameters:

            * ``"low"`` : the lower bound of the uniform distribution
              (default: -1.0)

            * ``"high"`` : the upper bound of the uniform distribution
              (default: 1.0)

        Returns
        -------
        n_rep : ``int``
            The number of representations.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """
        # Get the desired number of representations to be drawn
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations
        dim = options["dim"]

        # Get the bounds of the uniform distribution
        low = options.get("low", -1.0)
        high = options.get("high", 1.0)

        # Sample the values of the representations
        z = nn.Parameter(
            torch.empty(n_rep, dim, device=self.device).uniform_(low, high),
            requires_grad=True
        )
        
        # Return the number of representations, the dimensionality,
        # the values of the representations, and the options used
        return n_rep, dim, z, {"dist_name": "uniform", "low": low, "high": high}

    def _get_rep_from_laplace(self, options: Dict[str, Any]) -> Tuple[int, int, Tensor, Dict[str, Any]]:
        """Get the representations by sampling from a Laplace distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a Laplace distribution.

            Required parameters:
            * ``"n_samples"`` : the number of samples to draw
            * ``"dim"`` : the dimensionality of the representations

            Optional parameters:
            * ``"loc"`` : location parameter (default: 0.0)
            * ``"scale"`` : scale parameter (default: 1.0)

        Returns
        -------
        n_rep : ``int``
            The number of representations.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """
        # Get parameters
        n_rep = options["n_samples"]
        dim = options["dim"]
        loc = options.get("loc", 0.0)
        scale = options.get("scale", 1.0)

        # Sample from a uniform distribution then transform to Laplace
        uniform = torch.empty(n_rep, dim, device=self.device).uniform_(0, 1)
        # Convert uniform to Laplace using the inverse CDF
        sign = torch.sign(uniform - 0.5)
        z_val = loc - scale * sign * torch.log(1 - 2 * torch.abs(uniform - 0.5))
        
        # Create parameter
        z = nn.Parameter(z_val, requires_grad=True)
        
        return n_rep, dim, z, {"dist_name": "laplace", "loc": loc, "scale": scale}

    def _get_rep_from_student_t(self, options: Dict[str, Any]) -> Tuple[int, int, Tensor, Dict[str, Any]]:
        """Get the representations by sampling from a Student's t distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a Student's t distribution.

            Required parameters:
            * ``"n_samples"`` : the number of samples to draw
            * ``"dim"`` : the dimensionality of the representations

            Optional parameters:
            * ``"df"`` : degrees of freedom (default: 3.0)
            * ``"scale"`` : scale parameter (default: 1.0)

        Returns
        -------
        n_rep : ``int``
            The number of representations.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """
        # Get parameters
        n_rep = options["n_samples"]
        dim = options["dim"]
        df = options.get("df", 3.0)
        scale = options.get("scale", 1.0)

        # Sample from a Student's t distribution
        t_dist = torch.distributions.StudentT(df=df)
        samples = t_dist.sample((n_rep, dim)).to(self.device)
        
        # Scale the samples
        samples = samples * scale
        
        # Create parameter
        z = nn.Parameter(samples, requires_grad=True)
        
        return n_rep, dim, z, {
            "dist_name": "student_t",
            "df": df,
            "scale": scale
        }
    
    def _get_rep_from_cauchy(self, options: Dict[str, Any]) -> Tuple[int, int, Tensor, Dict[str, Any]]:
        """Get the representations by sampling from a Cauchy distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a Cauchy distribution.

            Required parameters:
            * ``"n_samples"`` : the number of samples to draw
            * ``"dim"`` : the dimensionality of the representations

            Optional parameters:
            * ``"scale"`` : scale parameter (default: 1.0)

        Returns
        -------
        n_rep : ``int``
            The number of representations.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """
        # Get parameters
        n_rep = options["n_samples"]
        dim = options["dim"]
        scale = options.get("scale", 1.0)

        # Sample from a Cauchy distribution
        cauchy_dist = torch.distributions.Cauchy(loc=0.0, scale=scale)
        samples = cauchy_dist.sample((n_rep, dim)).to(self.device)
        
        # Create parameter
        z = nn.Parameter(samples, requires_grad=True)
        
        return n_rep, dim, z, {
            "dist_name": "cauchy",
            "scale": scale
        }

    ########################### PROPERTIES ############################

    @property
    def n_rep(self) -> int:
        """The number of representations in the layer."""
        return self._n_rep

    @n_rep.setter
    def n_rep(self, value: int) -> None:
        """Raise an exception if the user tries to modify the value
        of ``n_rep`` after initialization.
        """
        errstr = (
            "The value of 'n_samples' is set at initialization and "
            "cannot be changed. If you want to change the number "
            "of representations in the layer, initialize a new "
            f"instance of '{self.__class__.__name__}'."
        )
        raise ValueError(errstr)

    @property
    def dim(self) -> int:
        """The dimensionality of the representations."""
        return self._dim

    @dim.setter
    def dim(self, value: int) -> None:
        """Raise an exception if the user tries to modify the value of
        ``dim`` after initialization.
        """
        errstr = (
            "The value of 'dim' is set at initialization and cannot "
            "be changed. If you want to change the dimensionality "
            "of the representations stored in the layer, initialize "
            f"a new instance of '{self.__class__.__name__}'."
        )
        raise ValueError(errstr)

    @property
    def options(self) -> Dict[str, Any]:
        """The dictionary of options used to generate the
        representations, if no values were passed when initializing
        the layer.
        """
        return self._options

    @options.setter
    def options(self, value: Dict[str, Any]) -> None:
        """Raise an exception if the user tries to modify the value of
        ``options`` after initialization.
        """
        errstr = (
            "The value of 'options' is set at initialization and "
            "cannot be changed. If you want to change the options "
            "used to generate the representations, initialize a "
            f"new instance of '{self.__class__.__name__}'."
        )
        raise ValueError(errstr)
    
    @property
    def z(self) -> Tensor:
        """The values of the representations."""
        return self._z

    @z.setter
    def z(self, value: Tensor) -> None:
        """Raise an exception if the user tries to modify the value of
        ``z`` after initialization.
        """
        errstr = (
            "The value of 'z' is set at initialization and cannot "
            "be changed. If you want to change the values of the "
            "representations stored in the layer, initialize a new "
            f"instance of '{self.__class__.__name__}'."
        )
        raise ValueError(errstr)

    ######################### PUBLIC METHODS ##########################

    def forward(self, ixs: Optional[Union[List[int], Tensor]] = None, 
                batch_size: Optional[int] = None) -> Tensor:
        """Forward pass - it returns the values of the representations.

        You can select a subset of representations to be returned using
        their numerical indexes.

        Parameters
        ----------
        ixs : ``list`` or ``Tensor``, optional
            The indexes of the samples whose representations should
            be returned. If not passed, all representations will be
            returned.
            
        batch_size : ``int``, optional
            If specified and ixs contains more than batch_size indices,
            processes the representations in batches for memory efficiency.

        Returns
        -------
        reps : ``torch.Tensor``
            A tensor containing the values of the representations for
            the samples of interest.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """
        # If no indexes were provided
        if ixs is None:
            # Return the values for all representations
            return self.z
        
        # If batch_size is specified and we have more indices than the batch size
        if batch_size is not None and len(ixs) > batch_size:
            # Process in batches for memory efficiency
            result_chunks = []
            for i in range(0, len(ixs), batch_size):
                batch_ixs = ixs[i:i+batch_size]
                result_chunks.append(self.z[batch_ixs])
            
            # Concatenate the results
            return torch.cat(result_chunks, dim=0)
        
        # Otherwise return representations for the specified indices
        return self.z[ixs]

    def to(self, device: Union[str, torch.device]) -> 'RepresentationLayer':
        """Move representations to specified device.
        
        Parameters
        ----------
        device : str or torch.device
            The device to move the representations to.
            
        Returns
        -------
        self : RepresentationLayer
            The representation layer itself, enabling method chaining.
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self._z = self._z.to(device)
        return super().to(device)
    
    def save(self, path: str) -> None:
        """Save the representations to a file.
        
        Parameters
        ----------
        path : str
            The path where to save the representations.
        """
        state_dict = {
            'z': self.z.detach().cpu(),
            'n_rep': self._n_rep,
            'dim': self._dim,
            'options': self._options
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[Union[str, torch.device]] = None) -> 'RepresentationLayer':
        """Load representations from a file.
        
        Parameters
        ----------
        path : str
            The path from which to load the representations.
            
        device : str or torch.device, optional
            The device where to load the representations.
            
        Returns
        -------
        rep_layer : RepresentationLayer
            The loaded representation layer.
        """
        state_dict = torch.load(path, map_location='cpu')
        rep_layer = cls(values=state_dict['z'], device=device)
        return rep_layer
    
    def regularization_loss(self, l1_weight: float = 0.0, 
                            l2_weight: float = 0.0) -> Tensor:
        """Calculate regularization loss for the representations.
        
        Parameters
        ----------
        l1_weight : float, default=0.0
            Weight for L1 regularization.
            
        l2_weight : float, default=0.0
            Weight for L2 regularization.
            
        Returns
        -------
        loss : Tensor
            The calculated regularization loss.
        """
        loss = torch.tensor(0.0, device=self.device)
        
        if l1_weight > 0:
            loss += l1_weight * torch.sum(torch.abs(self.z))
            
        if l2_weight > 0:
            loss += l2_weight * torch.sum(self.z ** 2)
            
        return loss
    
    def interpolate(self, idx1: int, idx2: int, steps: int = 10) -> Tensor:
        """Interpolate between two representations.
        
        Parameters
        ----------
        idx1 : int
            Index of the first representation.
            
        idx2 : int
            Index of the second representation.
            
        steps : int, default=10
            Number of interpolation steps.
            
        Returns
        -------
        interpolations : Tensor
            Tensor containing the interpolation steps.
        """
        start = self.z[idx1].detach()
        end = self.z[idx2].detach()
        
        alphas = torch.linspace(0, 1, steps, device=self.device)
        interpolations = []
        
        for alpha in alphas:
            interp = (1 - alpha) * start + alpha * end
            interpolations.append(interp)
            
        return torch.stack(interpolations)
    
    def mean_representation(self, indices: Union[List[int], Tensor]) -> Tensor:
        """Compute mean representation for given indices.
        
        Parameters
        ----------
        indices : list or Tensor
            Indices of the representations to average.
            
        Returns
        -------
        mean_rep : Tensor
            Mean representation.
        """
        return torch.mean(self.z[indices], dim=0, keepdim=True)
    
    def compute_statistics(self) -> Dict[str, Union[Tensor, float]]:
        """Compute basic statistics of representations.
        
        Returns
        -------
        stats : dict
            Dictionary containing the statistics.
        """
        with torch.no_grad():
            mean = torch.mean(self.z, dim=0)
            std = torch.std(self.z, dim=0)
            min_vals = torch.min(self.z, dim=0).values
            max_vals = torch.max(self.z, dim=0).values
            norms = torch.norm(self.z, dim=1)
            norm_mean = norms.mean().item()
            norm_std = norms.std().item()
        
        return {
            "mean": mean,
            "std": std,
            "min": min_vals,
            "max": max_vals,
            "norm_mean": norm_mean,
            "norm_std": norm_std
        }
    

    @classmethod
    def initialize_from_pca(cls, data: Tensor, n_components: Optional[int] = None, 
                            device: Optional[Union[str, torch.device]] = None) -> 'RepresentationLayer':
        """Initialize representations using PCA of input data.
        
        Parameters
        ----------
        data : Tensor
            The data to use for PCA initialization.
            
        n_components : int, optional
            Number of PCA components. If None, uses min(data.shape).
            
        device : str or torch.device, optional
            Device to place the representations on.
            
        Returns
        -------
        rep_layer : RepresentationLayer
            The initialized representation layer.
        """
        # Ensure data is a 2D tensor
        if isinstance(data, torch.Tensor):
            if len(data.shape) > 2:
                n_samples = data.shape[0]
                data = data.reshape(n_samples, -1)
        else:
            raise TypeError("Data must be a torch.Tensor")
        
        # Determine device
        if device is None:
            device = data.device
        
        # Determine number of components
        if n_components is None:
            n_components = min(data.shape)
        
        # Apply PCA directly in PyTorch
        pca = TorchPCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        
        # Create representation layer
        return cls(values=transformed_data, device=device)

    @classmethod
    def with_optimal_dim(cls, data: Tensor, min_dim: int = 2, max_dim: int = 100, 
                        criterion: str = 'explained_variance',
                        threshold: float = 0.95,
                        device: Optional[Union[str, torch.device]] = None) -> 'RepresentationLayer':
        """Create representation layer with automatically selected dimension.
        
        Parameters
        ----------
        data : Tensor
            The data to use for dimension selection.
            
        min_dim : int, default=2
            Minimum dimension to consider.
            
        max_dim : int, default=100
            Maximum dimension to consider.
            
        criterion : str, default='explained_variance'
            Criterion to use for selecting dimension.
            Options: 'explained_variance', 'elbow'
            
        threshold : float, default=0.95
            Threshold for explained variance criterion (between 0 and 1).
            
        device : str or torch.device, optional
            Device to place the representations on.
            
        Returns
        -------
        rep_layer : RepresentationLayer
            The initialized representation layer.
        """
        # Ensure data is a 2D tensor
        if isinstance(data, torch.Tensor):
            if len(data.shape) > 2:
                n_samples = data.shape[0]
                data = data.reshape(n_samples, -1)
        else:
            raise TypeError("Data must be a torch.Tensor")
        
        # Determine device
        if device is None:
            device = data.device
            
        # Apply PCA with the maximum number of components
        max_components = min(min(data.shape), max_dim)
        pca = TorchPCA(n_components=max_components)
        pca.fit(data)
        
        # Select optimal dimension based on criterion
        if criterion == 'explained_variance':
            # Find dimension where cumulative explained variance exceeds threshold
            cum_variance = torch.cumsum(pca.explained_variance_ratio_, dim=0)
            optimal_dim = torch.searchsorted(cum_variance, threshold).item() + 1
            optimal_dim = max(min_dim, min(optimal_dim, max_dim))
        
        elif criterion == 'elbow':
            # Use the elbow method (find where second derivative is maximized)
            explained_variance = pca.explained_variance_
            if len(explained_variance) > 2:
                # Calculate first differences
                d1 = explained_variance[:-1] - explained_variance[1:]
                # Calculate second differences
                d2 = d1[:-1] - d1[1:]
                # Find the elbow point
                optimal_dim = torch.argmax(d2).item() + 1
                optimal_dim = max(min_dim, min(optimal_dim, max_dim))
            else:
                optimal_dim = min_dim
        
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Use 'explained_variance' or 'elbow'")
        
        # Apply PCA with the optimal dimension
        transformed_data = pca.transform(data)[:, :optimal_dim]
        
        # Create representation layer
        return cls(values=transformed_data, device=device)
    



class RepresentationLayerOLD(nn.Module):
    
    """
    Class implementing a representation layer accumulating gradients.
    """


    ######################## PUBLIC ATTRIBUTE #########################


    # Set the available distributions to sample the representations
    # from.
    AVAILABLE_DISTS = ["normal"]


    ######################### INITIALIZATION ##########################

    
    def __init__(self, values=None, dist="normal", dist_options=None, device=None):

        """Initialize a representation layer.

        Parameters
        ----------
        values : ``torch.Tensor``, optional
            A tensor used to initialize the representations in
            the layer.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations in the tensor.

            * The second dimension has a length equal to the
              dimensionality of the representations.

            If the tensor is not passed, the representations will be
            initialized by sampling the distribution specified
            by ``dist``.

        dist : ``str``, {``"normal"``}, default: ``"normal"``
            The name of the distribution used to sample the
            representations, if no ``values`` are passed.

            By default, the distribution is a ``"normal"``
            distribution.

        dist_options : ``dict``, optional
            A dictionary containing the parameters to sample the
            representations from the distribution, if no ``values``
            are passed.

            For any distribution the following keys and associated
            parameters must be provided:

            * ``"n_samples"`` : the number of samples to draw from
              the distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the distribution.

            If ``dist`` is ``"normal"``, the dictionary must contain
            these additional key/value pairs:

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.
        """

        self.device = device or (values.device if values is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        
        # Initialize an instance of the 'nn.Module' class.
        super().__init__()


        # If a tensor of values was passed
        if values is not None:

            # Set the options used to initialize the representations
            # to an empty dictionary, since they have not been 
            # sampled from any distribution.
            self._options = {}

            # Get the number of representations, the
            # dimensionality of the representations, and the values
            # of the representations from the tensor.
            self._n_rep, self._dim, self._z = \
                self._get_rep_from_values(values = values)      
        
        # Otherwise
        else:

            # If the representations are to be sampled from a normal
            # distribution
            if dist == "normal":

                # Sample the representations from a normal
                # distribution.
                self._n_rep, self._dim, self._z, self._options = \
                    self._get_rep_from_normal(options = dist_options)

            # Otherwise
            else:

                # Raise an error.
                available_dists_str = \
                    ", ".join(f'{d}' for d in self.AVAILABLE_DISTS)
                errstr = \
                    f"Unsupported distribution '{dist}'. The only " \
                    "distributions from which it is possible to " \
                    "sample the representations are: " \
                    f"{available_dists_str}."
                raise ValueError(errstr)


    def _get_rep_from_values(self,
                             values):
        """Get the representations from a given tensor of values.

        Parameters
        ----------
        values : ``torch.Tensor``
            The tensor used to initialize the representations.

        Returns
        -------
        n_rep : ``int``
            The number of representations found in the input tensor.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """

        # Get the number of representations from the first dimension of
        # the tensor.
        n_rep = values.shape[0]
        
        # Get the dimensionality of the representations from the last
        # dimension of the tensor.
        dim = values.shape[-1]

        # Initialize a tensor with the representations.
        z = nn.Parameter(torch.zeros_like(values), 
                         requires_grad = True)

        # Fill the tensor with the given values.
        with torch.no_grad():
            z.copy_(values)

        # Return the number of representations, the dimensionality of
        # the representations, and the values of the representations.
        return n_rep, \
               dim, \
               z


    def _get_rep_from_normal(self,
                             options):
        """Get the representations by sampling from a normal
        distribution.

        Parameters
        ----------
        options : ``dict``
            A dictionary containing the parameters to sample the
            representations from a normal distribution.

            The dictionary must contains the following keys,
            associated with the corresponding parameters:

            * ``"n_samples"`` : the number of samples to draw from
              the normal distribution.

            * ``"dim"`` : the dimensionality of the representations
              to sample from the normal distribution.

            * ``"mean"`` : the mean of the normal distribution used
              to generate the representations.

            * ``"stddev"`` : the standard deviation of the normal
              distribution used to generate the representations.

        Returns
        -------
        n_rep : ``int``
            The number of representations found in the input tensor.

        dim : ``int``
            The dimensionality of the representations.

        rep : ``torch.Tensor``
            The values of the representations.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number of
              representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.

        options : ``dict``
            A dictionary containing the options used to initialize
            the representations.
        """

        # Get the desired number of representations to be drawn.
        n_rep = options["n_samples"]

        # Get the dimensionality of the desired representations.
        dim = options["dim"]

        # Get the mean of the normal distribution from which the
        # representations should be samples.
        mean = options["mean"]

        # Get the standard deviation of the normal distribution
        # from which the representations should be sampled.
        stddev = options["stddev"]

        # Get the values of the representations.
        z = \
            nn.Parameter(\
                torch.normal(mean,
                             stddev,
                             size = (n_rep, dim),
                             requires_grad = True))
        
        # Return the number of representations, the dimensionality of
        # the representations, the values of the representations,
        # and the options used to generate them.
        return n_rep, \
               dim, \
               z, \
               {"dist_name" : "normal",
                "mean" : mean,
                "stddev" : stddev}


    ########################### PROPERTIES ############################


    @property
    def n_rep(self):
        """The number of representations in the layer.
        """

        return self._n_rep


    @n_rep.setter
    def n_rep(self,
              value):
        """Raise an exception if the user tries to modify the value
        of ``n_rep`` after initialization.
        """
        
        errstr = \
            "The value of 'n_samples' is set at initialization and " \
            "cannot be changed. If you want to change the number " \
            "of representations in the layer, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def dim(self):
        """The dimensionality of the representations.
        """

        return self._dim


    @dim.setter
    def dim(self,
            value):
        """Raise an exception if the user tries to modify the value of
        ``dim`` after initialization.
        """
        
        errstr = \
            "The value of 'dim' is set at initialization and cannot " \
            "be changed. If you want to change the dimensionality " \
            "of the representations stored in the layer, initialize " \
            f"a new instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    @property
    def options(self):
        """The dictionary ot options used to generate the
        representations, if no values were passed when initializing
        the layer.
        """

        return self._options


    @options.setter
    def options(self,
                value):
        """Raise an exception if the user tries to modify the value of
        ``options`` after initialization.
        """
        
        errstr = \
            "The value of 'options' is set at initialization and " \
            "cannot be changed. If you want to change the options " \
            "used to generate the representations, initialize a " \
            f"new instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)
    

    @property
    def z(self):
        """The values of the representations.
        """

        return self._z


    @z.setter
    def z(self,
          value):
        """Raise an exception if the user tries to modify the value of
        ``z`` after initialization.
        """
        
        errstr = \
            "The value of 'z' is set at initialization and cannot " \
            "be changed. If you want to change the values of the " \
            "representations stored in the layer, initialize a new " \
            f"instance of '{self.__class__.__name__}'."
        raise ValueError(errstr)


    ######################### PUBLIC METHODS ##########################


    def forward(self,
                ixs = None):
        """Forward pass - it returns the values of the representations.

        You can select a subset of representations to be returned using
        their numerical indexes.

        Parameters
        ----------
        ixs : ``list``, optional
            The indexes of the samples whose representations should
            be returned. If not passed, all representations will be
            returned.

        Returns
        -------
        reps : ``torch.Tensor``
            A tensor containing the values of the representations for
            the samples of interest.

            This is a 2D tensor where:

            * The first dimension has a length equal to the number
              of representations.

            * The second dimension has a length equal to the
              dimensionality of the representations.
        """

        # If no indexes were provided
        if ixs is None:
            
            # Return the values for all representations.
            return self.z
        
        # Otherwise
        else:

            # Return the values for the representations of the
            # samples corresponding to the given indexes.
            return self.z[ixs]