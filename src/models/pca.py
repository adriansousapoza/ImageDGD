import torch

class TorchPCA:
    """PyTorch-based Principal Component Analysis (PCA).

    This class implements PCA directly in PyTorch, allowing for all operations
    to remain on GPU if available. The API is similar to sklearn.decomposition.PCA.

    Parameters
    ----------
    n_components : int or None, optional
        Number of components to keep. If None, all components are kept.
        By default, n_components=None.

    svd_solver : str, optional
        The SVD solver to use. Can be one of:
        - 'auto': chooses the most appropriate solver based on data shape
        - 'full': use full SVD with torch.linalg.svd
        - 'covariance_eigh': use eigenvalue decomposition of covariance matrix
        - 'randomized': use randomized SVD algorithm
        By default, svd_solver='auto'.
        
    whiten : bool, optional
        If True, the components are whitened to have unit variance.
        By default, whiten=False.

    random_state : int, optional
        Random state for reproducibility when using randomized SVD.
        By default, random_state=None.
    """
    
    def __init__(
        self,
        n_components=None,
        *,
        svd_solver="auto",
        whiten=False,
        random_state=None,
    ):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.whiten = whiten
        self.random_state = random_state
        
        # Attributes that will be set during fit
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
        self.noise_variance_ = None
    
    def fit(self, X):
        """Fit the PCA model to the data.
        
        Parameters
        ----------
        X : torch.Tensor
            Training data of shape (n_samples, n_features)
            
        Returns
        -------
        self : TorchPCA
            Returns self for method chaining.
        """
        n_samples, n_features = X.shape
        self.n_samples_ = n_samples
        self.n_features_ = n_features
        
        # Calculate mean and center data
        self.mean_ = torch.mean(X, dim=0, keepdim=True)
        X_centered = X - self.mean_
        
        # Choose SVD solver if 'auto' selected
        if self.svd_solver == "auto":
            self.svd_solver = self._choose_svd_solver(X)
        
        if self.svd_solver == "full":
            # Full SVD
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
            explained_variance = S**2 / (n_samples - 1)
            total_var = torch.sum(explained_variance)
            
        elif self.svd_solver == "covariance_eigh":
            # Eigenvalue decomposition of covariance matrix
            cov = torch.mm(X_centered.T, X_centered) / (n_samples - 1)
            eigenvals, eigenvecs = torch.linalg.eigh(cov)
            
            # Sort eigenvalues and eigenvectors in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Fix numerical errors
            eigenvals = torch.clamp(eigenvals, min=0.0)
            
            # Convert to SVD-like outputs
            explained_variance = eigenvals
            total_var = torch.sum(explained_variance)
            S = torch.sqrt(eigenvals * (n_samples - 1))
            Vt = eigenvecs.T
            U = None  # Not computed for efficiency
            
        elif self.svd_solver == "randomized":
            # Randomized SVD
            if self.n_components is None:
                n_components = min(n_samples, n_features)
            else:
                n_components = self.n_components
                
            U, S, Vt = self._randomized_svd(
                X_centered, n_components, random_state=self.random_state
            )
            explained_variance = S**2 / (n_samples - 1)
            total_var = torch.sum(X_centered**2) / (n_samples - 1)
        
        else:
            raise ValueError(f"Unknown SVD solver: {self.svd_solver}")
        
        # Calculate variance ratios
        explained_variance_ratio = explained_variance / total_var
        
        # Determine number of components to keep
        if self.n_components is None:
            n_components = min(n_samples, n_features)
        else:
            n_components = min(self.n_components, min(n_samples, n_features))
        
        # Store results
        self.components_ = Vt[:n_components]
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components]
        self.singular_values_ = S[:n_components]
        
        # Compute noise variance (the variance in the discarded dimensions)
        if n_components < min(n_samples, n_features):
            self.noise_variance_ = torch.mean(explained_variance[n_components:])
        else:
            self.noise_variance_ = torch.tensor(0.0, device=X.device)
        
        return self
    
    def transform(self, X):
        """Apply dimensionality reduction to X.
        
        Parameters
        ----------
        X : torch.Tensor
            Data to be transformed of shape (n_samples, n_features)
            
        Returns
        -------
        X_transformed : torch.Tensor
            Transformed data of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        X_centered = X - self.mean_
        X_transformed = torch.mm(X_centered, self.components_.T)
        
        if self.whiten:
            eps = 1e-8  # Small constant to avoid division by zero
            X_transformed /= torch.sqrt(self.explained_variance_ + eps)
            
        return X_transformed
    
    def fit_transform(self, X):
        """Fit the model and apply the dimensionality reduction on X.
        
        Parameters
        ----------
        X : torch.Tensor
            Training data of shape (n_samples, n_features)
            
        Returns
        -------
        X_transformed : torch.Tensor
            Transformed data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Transform data back to original space.
        
        Parameters
        ----------
        X : torch.Tensor
            Data in transformed space of shape (n_samples, n_components)
            
        Returns
        -------
        X_original : torch.Tensor
            Data in original space of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        if self.whiten:
            eps = 1e-8  # Small constant to avoid division by zero
            X_transformed = X * torch.sqrt(self.explained_variance_ + eps)
        else:
            X_transformed = X
            
        X_original = torch.mm(X_transformed, self.components_) + self.mean_
        return X_original
    
    def _choose_svd_solver(self, X):
        """Choose the SVD solver based on the input shape and n_components."""
        n_samples, n_features = X.shape
        
        if n_features <= 1000 and n_samples >= 10 * n_features:
            return "covariance_eigh"
        elif max(n_samples, n_features) <= 500:
            return "full"
        elif (isinstance(self.n_components, int) and 
              1 <= self.n_components < 0.8 * min(n_samples, n_features)):
            return "randomized"
        else:
            return "full"
    
    def _randomized_svd(self, X, n_components, n_oversamples=10, n_iter=4, random_state=None):
        """Compute randomized SVD.
        
        Implementation based on the algorithm of Halko et al.
        """
        n_samples, n_features = X.shape
        
        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
        
        # Step 1: Find a subspace that captures most of the action
        n_random = n_components + n_oversamples
        random_matrix = torch.randn(n_features, n_random, device=X.device)
        Q = torch.mm(X, random_matrix)
        
        # Step 2: Power iterations to increase accuracy
        for _ in range(n_iter):
            Q = torch.mm(X, torch.mm(X.T, Q))
            Q, _ = torch.linalg.qr(Q)
        
        # Step 3: Compute SVD on the smaller matrix B = Q^T * X
        B = torch.mm(Q.T, X)
        Uhat, S, Vt = torch.linalg.svd(B, full_matrices=False)
        U = torch.mm(Q, Uhat)
        
        # Return only the first n_components
        return U[:, :n_components], S[:n_components], Vt[:n_components]
