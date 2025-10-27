import os
import os.path as osp
from typing import Tuple, Dict, Any
import numpy as np    
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import sys

class IndexedDataset(torch.utils.data.Dataset):
    """
    Wrap a dataset to also return its *global* index as the third item.
    Works with Subset (the index will still be the original dataset index).
    """
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i  # i is the global index in the wrapped dataset


def get_loss_collecting_dataloaders(**kwargs):
    """
    Returns (train_loader, eval_train_loader, validation_loader, test_loader, metadata)

    metadata contains:
      - 'train_indices': list of original dataset indices used for the train subset (order aligns with eval_train)
      - 'val_indices':   list of original dataset indices used for the val subset
      - 'index_to_pos':  dict mapping original index -> position in train_indices
      - 'train_len':     length of the train subset
    """

    data_path  = kwargs.get('data_path', './Data')
    dataset    = str(kwargs.get('dataset', 'CIFAR100')).lower()

    workers        = kwargs.get('workers', 4)
    workers        = 4   if workers is None else int(workers)
    batch_size     = kwargs.get('batch_size', 128)
    batch_size     = 128 if batch_size is None else int(batch_size)
    test_batch_size= kwargs.get('test_batch_size', 256)
    test_batch_size= 256 if test_batch_size is None else int(test_batch_size)
    manual_seed    = kwargs.get('manual_seed', 42)
    manual_seed    = 42  if manual_seed is None else int(manual_seed)

    torch.manual_seed(manual_seed)
    g = torch.Generator().manual_seed(manual_seed)  # for per-class shuffles

    ds = dataset.lower()

    if ds == 'cifar100':
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408),
                                         (0.2675, 0.2565, 0.2761))
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        eval_tf = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        base_train = datasets.CIFAR100(root=osp.join(data_path, 'cifar'),
                                       train=True, download=True, transform=train_tf)
        base_eval  = datasets.CIFAR100(root=osp.join(data_path, 'cifar'),
                                       train=True, download=True, transform=eval_tf)
        base_test  = datasets.CIFAR100(root=osp.join(data_path, 'cifar'),
                                       train=False, download=True, transform=eval_tf)

        # Wrap to emit indices
        train_wrapped = IndexedDataset(base_train)
        eval_wrapped  = IndexedDataset(base_eval)

        val_ratio   = 0.10
        num_classes = 100
        # CIFAR100 stores labels in .targets
        targets = torch.as_tensor(base_eval.targets, dtype=torch.long)
        train_indices, val_indices = [], []
        for c in range(num_classes):
            cls_idxs  = torch.nonzero(targets == c, as_tuple=False).squeeze(1)
            perm      = cls_idxs[torch.randperm(len(cls_idxs), generator=g)]
            split_at  = int(len(perm) * (1.0 - val_ratio))
            train_indices.extend(perm[:split_at].tolist())
            val_indices.extend(perm[split_at:].tolist())

        train_subset      = Subset(train_wrapped, train_indices)
        eval_train_subset = Subset(eval_wrapped,  train_indices)  # same order as train_indices
        validation_subset = Subset(eval_wrapped,  val_indices)
        testset           = base_test

    elif ds == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        eval_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_dir = osp.join(data_path, 'imagenet', 'train')
        val_dir   = osp.join(data_path, 'imagenet', 'val')

        base_train = datasets.ImageFolder(train_dir, transform=train_tf)
        base_eval  = datasets.ImageFolder(train_dir, transform=eval_tf)
        base_test  = datasets.ImageFolder(val_dir,   transform=eval_tf)

        train_wrapped = IndexedDataset(base_train)
        eval_wrapped  = IndexedDataset(base_eval)

        val_ratio = 0.01
        # Robustly get class indices for each sample
        if hasattr(base_eval, 'targets') and len(base_eval.targets) == len(base_eval.samples):
            targets_np = np.asarray(base_eval.targets, dtype=np.int64)
        else:
            targets_np = np.asarray([cls for _, cls in base_eval.samples], dtype=np.int64)
        targets = torch.as_tensor(targets_np, dtype=torch.long)
        num_classes = len(base_eval.classes)

        train_indices, val_indices = [], []
        for c in range(num_classes):
            cls_idxs  = torch.nonzero(targets == c, as_tuple=False).squeeze(1)
            if len(cls_idxs) == 0:
                continue
            perm      = cls_idxs[torch.randperm(len(cls_idxs), generator=g)]
            split_at  = int(len(perm) * (1.0 - val_ratio))
            train_indices.extend(perm[:split_at].tolist())
            val_indices.extend(perm[split_at:].tolist())

        train_subset      = Subset(train_wrapped, train_indices)
        eval_train_subset = Subset(eval_wrapped,  train_indices)
        validation_subset = Subset(eval_wrapped,  val_indices)
        testset           = base_test

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented in dataset_utils.py")

    train_loader       = DataLoader(train_subset,      batch_size=batch_size,     shuffle=True,
                                    num_workers=workers, pin_memory=True)
    eval_train_loader  = DataLoader(eval_train_subset, batch_size=test_batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=True)
    validation_loader  = DataLoader(validation_subset, batch_size=test_batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=True)
    test_loader        = DataLoader(testset,           batch_size=test_batch_size, shuffle=False,
                                    num_workers=workers, pin_memory=True)

    index_to_pos = {idx: pos for pos, idx in enumerate(train_indices)}
    meta = {
        'train_indices': train_indices,
        'val_indices':   val_indices,
        'index_to_pos':  index_to_pos,
        'train_len':     len(train_indices),
    }

    return train_loader, eval_train_loader, validation_loader, test_loader, meta


def get_dataloaders_for_coresets(**kwargs):

    data_path = kwargs.get('data_path', './Data')
    dataset = str(kwargs.get('dataset')).lower()
    workers = kwargs.get('workers', 4)
    workers = 4 if workers is None else int(workers)
    batch_size = kwargs.get('batch_size', 128)
    batch_size = 128 if batch_size is None else int(batch_size)
    scores = kwargs.get('scores')  # dict: class_id -> indices (or (idx,score) pairs)
    topk_per_class = int(kwargs.get('topk_per_class', 100))

    if scores is None:
        raise ValueError("get_dataloaders_for_coresets: 'scores' dict is required.")

    coreset_indices = []
    for k in scores.keys():
        # keys may be "0" or 0; normalize lookups both ways
        entry = scores.get(k, None)
        if entry is None and isinstance(k, str):
            entry = scores.get(int(k), None)
        if entry is None and isinstance(k, int):
            entry = scores.get(str(k), None)
        if entry is None:
            continue

        arr = np.asarray(entry, dtype=object)

        # Case A: 1-D array of indices
        if arr.ndim == 1 and (len(arr) == 0 or not isinstance(arr[0], (list, tuple, np.ndarray))):
            idxs = arr[:topk_per_class].astype(int).tolist()

        # Case B: list/array of pairs (index, score) or multi-col where col0=index
        elif (arr.ndim == 2 and arr.shape[1] >= 1) or (len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray))):
            try:
                idxs = np.asarray([row[0] for row in arr[:topk_per_class]], dtype=int).tolist()
            except Exception:
                # handle numpy 2D directly
                idxs = np.asarray(arr[:topk_per_class, 0], dtype=int).tolist()
        else:
            raise ValueError(f"Unsupported scores format for class key {k}: shape {arr.shape}")

        coreset_indices.extend(idxs)

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Use same root convention as the rest of the repo
        original_trainset = datasets.CIFAR100(
            root=osp.join(data_path, 'cifar'), train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(
            root=osp.join(data_path, 'cifar'), train=False, download=True, transform=transform_test)

        coreset = Subset(original_trainset, coreset_indices)

    elif dataset == 'imagenet':
        traindir = os.path.join(data_path, 'imagenet', 'train')
        valdir   = os.path.join(data_path, 'imagenet', 'val')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        original_trainset = datasets.ImageFolder(root=traindir, transform=train_transform)
        testset           = datasets.ImageFolder(root=valdir,   transform=test_transform)
        coreset = Subset(original_trainset, coreset_indices)

    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented.")

    train_loader = DataLoader(coreset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=workers,
                             pin_memory=True)

    return train_loader, test_loader