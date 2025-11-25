"""
Data loading and preprocessing utilities for ImageDGD.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from omegaconf import DictConfig


class IndexedDataset(Dataset):
    """
    Dataset wrapper that provides indices along with data and labels.
    Supports subsetting for debugging purposes.
    """
    
    def __init__(self, dataset: Dataset, subset_fraction: float = 1.0):
        """
        Wrap a dataset with indices
        
        Parameters:
        ----------
        dataset: The original dataset
        subset_fraction: Fraction of data to use (1.0 = full dataset, < 1.0 = subset)
        """
        self.dataset = dataset
        
        if subset_fraction < 1.0:
            # Create a subset of indices
            total_size = len(dataset)
            subset_size = int(total_size * subset_fraction)
            
            # Create random indices ensuring we get samples from all classes
            all_indices = list(range(total_size))
            np.random.shuffle(all_indices)
            self.indices = all_indices[:subset_size]
        else:
            # Use all indices
            self.indices = list(range(len(dataset)))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
        # Map the index to the original dataset index
        orig_index = self.indices[index]
        data, target = self.dataset[orig_index]
        return orig_index, data, target


def get_transform(transform_config: List[str]) -> transforms.Compose:
    """
    Create torchvision transforms from configuration.
    
    Parameters:
    ----------
    transform_config: List of transform names
    
    Returns:
    -------
    Composed transforms
    """
    transform_list = []
    
    for transform_name in transform_config:
        if transform_name == "ToTensor":
            transform_list.append(transforms.ToTensor())
        elif transform_name == "Normalize":
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        elif transform_name == "RandomHorizontalFlip":
            transform_list.append(transforms.RandomHorizontalFlip(0.5))
        else:
            raise ValueError(f"Unknown transform: {transform_name}")
    
    return transforms.Compose(transform_list)


def create_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Create train and test dataloaders based on configuration.
    
    Parameters:
    ----------
    config: Data configuration
    
    Returns:
    -------
    Tuple of (train_loader, test_loader, class_names)
    """
    data_config = config.data
    
    # Create transforms
    train_transform = get_transform(data_config.transforms.train)
    test_transform = get_transform(data_config.transforms.test)
    
    # Load datasets
    if data_config.dataset_name == "FashionMNIST":
        train_dataset = datasets.FashionMNIST(
            root=data_config.root_dir, 
            train=True, 
            download=data_config.download, 
            transform=train_transform
        )
        test_dataset = datasets.FashionMNIST(
            root=data_config.root_dir, 
            train=False, 
            download=data_config.download, 
            transform=test_transform
        )
        # Get actual FashionMNIST class names (in correct order 0-9)
        actual_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError(f"Unknown dataset: {data_config.dataset_name}")
    
    # Combine both datasets to create a subset from total data
    from torch.utils.data import ConcatDataset
    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    total_size = len(combined_dataset)
    
    # Get total subset fraction and test split ratio
    total_subset_fraction = getattr(data_config, 'total_subset_fraction', getattr(data_config, 'subset_fraction', 1.0))
    test_split = getattr(data_config, 'test_split', 0.5)
    
    # Calculate subset size
    subset_size = int(total_size * total_subset_fraction)
    
    # Create random indices for the subset
    np.random.seed(config.random_seed)
    all_indices = list(range(total_size))
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]
    
    # Split subset into train and test
    split_point = int(len(subset_indices) * (1 - test_split))
    train_indices = subset_indices[:split_point]
    test_indices = subset_indices[split_point:]
    
    # Create custom datasets with the split indices
    class SubsetDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            orig_idx = self.indices[idx]
            data, target = self.dataset[orig_idx]
            return orig_idx, data, target
    
    indexed_train_dataset = SubsetDataset(combined_dataset, train_indices)
    indexed_test_dataset = SubsetDataset(combined_dataset, test_indices)
    
    # Adjust batch size if using a small dataset
    batch_size = data_config.batch_size
    if total_subset_fraction < 1.0:
        batch_size = min(batch_size, max(32, len(indexed_train_dataset) // 10))
        print(f"Using {total_subset_fraction*100:.0f}% of total data with {test_split*100:.0f}/{(1-test_split)*100:.0f} test/train split")
        print(f"Adjusted batch size: {batch_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        indexed_train_dataset,
        batch_size=batch_size,
        shuffle=data_config.shuffle_train,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    test_loader = DataLoader(
        indexed_test_dataset,
        batch_size=batch_size,
        shuffle=data_config.shuffle_test,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    # Use actual dataset class names (in correct label order) instead of config
    class_names = actual_class_names
    
    print(f'Train dataset: {len(indexed_train_dataset)} samples ({len(indexed_train_dataset)/total_size*100:.1f}% of total)')
    print(f'Test dataset: {len(indexed_test_dataset)} samples ({len(indexed_test_dataset)/total_size*100:.1f}% of total)')
    print(f'Total original data: {total_size} samples')
    sample_data, sample_target = combined_dataset[0]
    print(f'Image shape: {sample_data.shape}')
    print(f'Image size: {sample_data.numel()}')
    
    return train_loader, test_loader, class_names


def get_sample_batches(train_loader: DataLoader, test_loader: DataLoader, device: torch.device, 
                      n_per_class: int = 5, n_classes: int = 10) -> Tuple:
    """
    Get sample batches for visualization, ensuring coverage of all classes.
    
    Parameters:
    ----------
    train_loader: Training data loader
    test_loader: Test data loader  
    device: Device to move data to
    n_per_class: Number of samples per class to collect
    n_classes: Total number of classes
    
    Returns:
    -------
    Tuple of sample batch data with guaranteed class coverage
    """
    def collect_class_samples(loader, n_per_class, n_classes):
        """Collect n_per_class samples for each class."""
        class_samples = {i: {'indices': [], 'images': [], 'labels': []} for i in range(n_classes)}
        
        # Iterate through batches until we have enough samples per class
        for indices, images, labels in loader:
            for idx, img, label in zip(indices, images, labels):
                label_val = label.item()
                if len(class_samples[label_val]['indices']) < n_per_class:
                    class_samples[label_val]['indices'].append(idx)
                    class_samples[label_val]['images'].append(img)
                    class_samples[label_val]['labels'].append(label)
            
            # Check if we have enough samples for all classes
            if all(len(class_samples[i]['indices']) >= n_per_class for i in range(n_classes)):
                break
        
        # Combine all samples
        all_indices = []
        all_images = []
        all_labels = []
        
        for class_idx in range(n_classes):
            all_indices.extend(class_samples[class_idx]['indices'])
            all_images.extend(class_samples[class_idx]['images'])
            all_labels.extend(class_samples[class_idx]['labels'])
        
        return (
            torch.stack([torch.tensor(i) for i in all_indices]),
            torch.stack(all_images),
            torch.stack(all_labels)
        )
    
    # Collect samples from train and test loaders
    indices_train, images_train, labels_train = collect_class_samples(train_loader, n_per_class, n_classes)
    indices_test, images_test, labels_test = collect_class_samples(test_loader, n_per_class, n_classes)
    
    return (
        indices_train.to(device), images_train.to(device), labels_train.to(device),
        indices_test.to(device), images_test.to(device), labels_test.to(device)
    )


def save_sample_batches(sample_data: Tuple, save_path: str):
    """
    Save sample batches to disk for later inference.
    
    Parameters:
    ----------
    sample_data: Tuple of (indices_train, images_train, labels_train, 
                           indices_test, images_test, labels_test)
    save_path: Path to save the samples
    """
    indices_train, images_train, labels_train, indices_test, images_test, labels_test = sample_data
    
    # Move to CPU for saving
    torch.save({
        'indices_train': indices_train.cpu(),
        'images_train': images_train.cpu(),
        'labels_train': labels_train.cpu(),
        'indices_test': indices_test.cpu(),
        'images_test': images_test.cpu(),
        'labels_test': labels_test.cpu(),
    }, save_path)
    
    print(f"Saved sample batches to {save_path}")
    print(f"  Train samples: {len(images_train)}")
    print(f"  Test samples: {len(images_test)}")


def load_sample_batches(load_path: str, device: torch.device) -> Tuple:
    """
    Load sample batches from disk.
    
    Parameters:
    ----------
    load_path: Path to load the samples from
    device: Device to move data to
    
    Returns:
    -------
    Tuple of sample batch data
    """
    data = torch.load(load_path, map_location=device)
    
    return (
        data['indices_train'].to(device),
        data['images_train'].to(device),
        data['labels_train'].to(device),
        data['indices_test'].to(device),
        data['images_test'].to(device),
        data['labels_test'].to(device)
    )
