"""
Data loading and preprocessing utilities for ImageDGD.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
        self.dataset = dataset

        if subset_fraction < 1.0:
            total_size = len(dataset)
            subset_size = int(total_size * subset_fraction)
            all_indices = list(range(total_size))
            np.random.shuffle(all_indices)
            self.indices = all_indices[:subset_size]
        else:
            self.indices = list(range(len(dataset)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int]:
        orig_index = self.indices[index]
        data, target = self.dataset[orig_index]
        return orig_index, data, target


def get_transform(transform_config: List[str]) -> transforms.Compose:
    """Create torchvision transforms from configuration."""
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


class SubsetDataset(Dataset):
    """
    Wraps a dataset with an explicit index list, remapping to sequential
    0..N-1 indices so they line up with a RepresentationLayer's rows.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        orig_idx = self.indices[idx]
        data, target = self.dataset[orig_idx]
        # Return sequential index (idx) instead of orig_idx, so it matches
        # the corresponding RepresentationLayer's row for this split.
        return idx, data, target


def _build_combined_dataset(data_config: DictConfig, transform: transforms.Compose) -> ConcatDataset:
    """Pool FashionMNIST's native train+test physical partitions into one
    dataset, both read through the same transform pipeline. Caller has
    already validated data_config.dataset_name == "FashionMNIST"."""
    train_part = datasets.FashionMNIST(
        root=data_config.root_dir, train=True, download=data_config.download, transform=transform
    )
    test_part = datasets.FashionMNIST(
        root=data_config.root_dir, train=False, download=data_config.download, transform=transform
    )
    return ConcatDataset([train_part, test_part])


def create_dataloaders(config: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders based on configuration.

    Pools FashionMNIST's native train+test partitions (70000 images total),
    shuffles with `config.random_seed`, and slices into three splits: test
    first, then val, then train from the remainder. This order is arbitrary
    but must stay fixed, since `dgd_test_inference.ipynb` re-derives the same
    test split independently by re-running this function with the same
    `random_seed`, rather than reading indices from a saved file.

    Parameters:
    ----------
    config: Full configuration (uses config.data and config.random_seed)

    Returns:
    -------
    Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    data_config = config.data

    if data_config.dataset_name != "FashionMNIST":
        raise ValueError(f"Unknown dataset: {data_config.dataset_name}")
    actual_class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_transform = get_transform(data_config.transforms.train)
    val_transform = get_transform(data_config.transforms.val)
    test_transform = get_transform(data_config.transforms.test)

    # One pooled dataset per transform pipeline, so each split's images are
    # read through that split's own (non-)augmented transform regardless of
    # which physical FashionMNIST partition they originated from.
    combined_train = _build_combined_dataset(data_config, train_transform)
    combined_val = _build_combined_dataset(data_config, val_transform)
    combined_test = _build_combined_dataset(data_config, test_transform)
    total_size = len(combined_train)

    total_subset_fraction = getattr(data_config, 'total_subset_fraction', getattr(data_config, 'subset_fraction', 1.0))
    val_split = getattr(data_config, 'val_split', 0.1)
    test_split = getattr(data_config, 'test_split', 0.1)

    subset_size = int(total_size * total_subset_fraction)

    np.random.seed(config.random_seed)
    all_indices = list(range(total_size))
    np.random.shuffle(all_indices)
    subset_indices = all_indices[:subset_size]

    n = len(subset_indices)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    test_indices = subset_indices[:n_test]
    val_indices = subset_indices[n_test:n_test + n_val]
    train_indices = subset_indices[n_test + n_val:]

    indexed_train_dataset = SubsetDataset(combined_train, train_indices)
    indexed_val_dataset = SubsetDataset(combined_val, val_indices)
    indexed_test_dataset = SubsetDataset(combined_test, test_indices)

    batch_size = data_config.batch_size
    if total_subset_fraction < 1.0:
        batch_size = min(batch_size, max(32, len(indexed_train_dataset) // 10))
        print(f"Using {total_subset_fraction*100:.0f}% of total data with "
              f"{(1-val_split-test_split)*100:.0f}/{val_split*100:.0f}/{test_split*100:.0f} train/val/test split")
        print(f"Adjusted batch size: {batch_size}")

    train_loader = DataLoader(
        indexed_train_dataset, batch_size=batch_size, shuffle=data_config.shuffle_train,
        num_workers=data_config.num_workers, pin_memory=data_config.pin_memory
    )
    val_loader = DataLoader(
        indexed_val_dataset, batch_size=batch_size, shuffle=data_config.shuffle_val,
        num_workers=data_config.num_workers, pin_memory=data_config.pin_memory
    )
    test_loader = DataLoader(
        indexed_test_dataset, batch_size=batch_size, shuffle=data_config.shuffle_test,
        num_workers=data_config.num_workers, pin_memory=data_config.pin_memory
    )

    class_names = actual_class_names

    print(f'Train dataset: {len(indexed_train_dataset)} samples ({len(indexed_train_dataset)/total_size*100:.1f}% of total)')
    print(f'Val dataset: {len(indexed_val_dataset)} samples ({len(indexed_val_dataset)/total_size*100:.1f}% of total)')
    print(f'Test dataset: {len(indexed_test_dataset)} samples ({len(indexed_test_dataset)/total_size*100:.1f}% of total)')
    print(f'Total original data: {total_size} samples')
    sample_data, sample_target = combined_train[0]
    print(f'Image shape: {sample_data.shape}')
    print(f'Image size: {sample_data.numel()}')

    return train_loader, val_loader, test_loader, class_names


def collect_all_labels(loader: DataLoader) -> torch.Tensor:
    """Collect every sample's label, indexed to match RepresentationLayer rows
    (labels[i] is the label of the sample whose representation is rep.z[i])."""
    n_samples = len(loader.dataset)
    labels = torch.zeros(n_samples, dtype=torch.long)
    for index, _, labels_batch in loader:
        labels[index] = labels_batch
    return labels


def collect_class_samples(loader: DataLoader, n_per_class: int, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect up to n_per_class samples for each class from a single loader."""
    class_samples = {i: {'indices': [], 'images': [], 'labels': []} for i in range(n_classes)}

    for indices, images, labels in loader:
        for idx, img, label in zip(indices, images, labels):
            label_val = label.item()
            if len(class_samples[label_val]['indices']) < n_per_class:
                class_samples[label_val]['indices'].append(idx)
                class_samples[label_val]['images'].append(img)
                class_samples[label_val]['labels'].append(label)
        if all(len(class_samples[i]['indices']) >= n_per_class for i in range(n_classes)):
            break

    all_indices, all_images, all_labels = [], [], []
    for class_idx in range(n_classes):
        all_indices.extend(class_samples[class_idx]['indices'])
        all_images.extend(class_samples[class_idx]['images'])
        all_labels.extend(class_samples[class_idx]['labels'])

    return (
        torch.stack([torch.tensor(i) for i in all_indices]),
        torch.stack(all_images),
        torch.stack(all_labels)
    )


def get_sample_batches(train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                      n_per_class: int = 5, n_classes: int = 10) -> Tuple:
    """
    Get sample batches for visualization, ensuring coverage of all classes.

    Parameters:
    ----------
    train_loader: Training data loader
    val_loader: Validation data loader
    device: Device to move data to
    n_per_class: Number of samples per class to collect
    n_classes: Total number of classes

    Returns:
    -------
    Tuple of (indices_train, images_train, labels_train, indices_val, images_val, labels_val)
    """
    indices_train, images_train, labels_train = collect_class_samples(train_loader, n_per_class, n_classes)
    indices_val, images_val, labels_val = collect_class_samples(val_loader, n_per_class, n_classes)

    return (
        indices_train.to(device), images_train.to(device), labels_train.to(device),
        indices_val.to(device), images_val.to(device), labels_val.to(device)
    )


def save_sample_batches(sample_data: Tuple, save_path: str):
    """
    Save sample batches to disk for later inference.

    Parameters:
    ----------
    sample_data: Tuple of (indices_train, images_train, labels_train,
                           indices_val, images_val, labels_val)
    save_path: Path to save the samples
    """
    indices_train, images_train, labels_train, indices_val, images_val, labels_val = sample_data

    torch.save({
        'indices_train': indices_train.cpu(),
        'images_train': images_train.cpu(),
        'labels_train': labels_train.cpu(),
        'indices_val': indices_val.cpu(),
        'images_val': images_val.cpu(),
        'labels_val': labels_val.cpu(),
    }, save_path)

    print(f"Saved sample batches to {save_path}")
    print(f"  Train samples: {len(images_train)}")
    print(f"  Val samples: {len(images_val)}")


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
        data['indices_val'].to(device),
        data['images_val'].to(device),
        data['labels_val'].to(device)
    )
