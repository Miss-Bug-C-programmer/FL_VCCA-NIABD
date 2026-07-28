from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
import os
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F


class TensorImageDataset(torch.utils.data.Dataset):
    """Simple tensor-backed image dataset with torchvision-like targets field."""

    def __init__(self, data: torch.Tensor, targets: torch.Tensor):
        self.data = data
        self.targets = [int(x) for x in targets.view(-1).tolist()]

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, index: int):
        x = self.data[index]
        y = int(self.targets[index])
        return x, y


class ProxyInputDataset(torch.utils.data.Dataset):
    """Proxy view that exposes inputs without exposing admission labels."""

    def __init__(self, dataset, indices: List[int]):
        self._dataset = dataset
        self.indices = tuple(int(index) for index in indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample = self._dataset[self.indices[int(index)]]
        if not isinstance(sample, (tuple, list)) or not sample:
            raise ValueError("Proxy dataset samples must contain an input.")
        return sample[0]


@dataclass(frozen=True)
class FederatedDataPlan:
    """Deterministic process-runtime partition and proxy identity."""

    dataset_path: str
    dataset_name: str
    seed: int
    client_indices: Tuple[Tuple[int, ...], ...]
    proxy_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    partition_scheme: str
    label_skew_classes: int
    quantity_skew_alpha: float
    batch_size: int
    transform_identity: str
    proxy_version: str

    @property
    def num_clients(self) -> int:
        return len(self.client_indices)


def _normalize_femnist_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 3:
        # [N,H,W]
        x = x.unsqueeze(1)
    elif x.ndim == 4 and x.shape[-1] == 1:
        # [N,H,W,1] -> [N,1,H,W]
        x = x.permute(0, 3, 1, 2).contiguous()
    if x.ndim != 4:
        raise ValueError(f"Unsupported FEMNIST tensor shape: {tuple(x.shape)}")

    x = x.float()
    if float(x.max().item()) > 1.0:
        x = x / 255.0

    # Keep model stack compatible: convert grayscale FEMNIST to 3x32x32.
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    if x.shape[2] != 32 or x.shape[3] != 32:
        x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)

    x = (x - 0.5) / 0.5
    return x.contiguous()


def _load_femnist_split(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if path.endswith('.npz'):
        obj = np.load(path, allow_pickle=True)
        if 'x' not in obj or 'y' not in obj:
            raise ValueError(f"FEMNIST npz file must include keys 'x' and 'y': {path}")
        x = torch.from_numpy(np.asarray(obj['x']))
        y = torch.from_numpy(np.asarray(obj['y']))
        return x, y

    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return torch.as_tensor(obj[0]), torch.as_tensor(obj[1])
    if isinstance(obj, dict):
        x = None
        y = None
        for k in ('x', 'data', 'images'):
            if k in obj:
                x = obj[k]
                break
        for k in ('y', 'targets', 'labels'):
            if k in obj:
                y = obj[k]
                break
        if x is None or y is None:
            raise ValueError(f"FEMNIST pt dict missing x/y keys in {path}. Supported keys: x|data|images and y|targets|labels")
        return torch.as_tensor(x), torch.as_tensor(y)
    raise ValueError(f"Unsupported FEMNIST split file format: {path}")


def _load_femnist_datasets(dataset_path: str) -> Tuple[TensorImageDataset, TensorImageDataset]:
    candidates = [
        ('femnist_train.pt', 'femnist_test.pt'),
        ('train.pt', 'test.pt'),
        (os.path.join('femnist', 'train.pt'), os.path.join('femnist', 'test.pt')),
        ('femnist_train.npz', 'femnist_test.npz'),
        ('train.npz', 'test.npz'),
        (os.path.join('femnist', 'train.npz'), os.path.join('femnist', 'test.npz')),
    ]
    train_path = None
    test_path = None
    for tr, te in candidates:
        tr_abs = os.path.join(dataset_path, tr)
        te_abs = os.path.join(dataset_path, te)
        if os.path.exists(tr_abs) and os.path.exists(te_abs):
            train_path, test_path = tr_abs, te_abs
            break
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "FEMNIST files not found. Expected one of: "
            "{femnist_train.pt,femnist_test.pt}, {train.pt,test.pt}, "
            "{femnist/train.pt,femnist/test.pt}, or npz equivalents."
        )

    train_x, train_y = _load_femnist_split(train_path)
    test_x, test_y = _load_femnist_split(test_path)
    train_ds = TensorImageDataset(_normalize_femnist_tensor(train_x), torch.as_tensor(train_y).long().view(-1))
    test_ds = TensorImageDataset(_normalize_femnist_tensor(test_x), torch.as_tensor(test_y).long().view(-1))
    return train_ds, test_ds


def _extract_targets(dataset) -> List[int]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets; unsupported in current prototype.")
    return [int(x) for x in targets]


def _iid_partition(indices: List[int], num_clients: int, rng: random.Random) -> List[List[int]]:
    shuffled = list(indices)
    rng.shuffle(shuffled)
    chunks = [[] for _ in range(num_clients)]
    for idx, sample_idx in enumerate(shuffled):
        chunks[idx % num_clients].append(sample_idx)
    return chunks


def _label_skew_partition(targets: List[int], indices: List[int], num_clients: int, classes_per_client: int, seed: int) -> List[List[int]]:
    rng = random.Random(int(seed) + 101)
    all_classes = sorted(set(targets[i] for i in indices))
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        by_class[int(targets[idx])].append(idx)
    for idxs in by_class.values():
        rng.shuffle(idxs)

    client_classes = []
    for client_id in range(num_clients):
        chosen = []
        while len(chosen) < min(classes_per_client, len(all_classes)):
            cls = all_classes[(client_id + len(chosen)) % len(all_classes)]
            if cls not in chosen:
                chosen.append(cls)
        rng.shuffle(chosen)
        client_classes.append(set(chosen))

    partitions = [[] for _ in range(num_clients)]
    for cls in all_classes:
        eligible = [cid for cid in range(num_clients) if cls in client_classes[cid]]
        if not eligible:
            eligible = list(range(num_clients))
        class_indices = list(by_class[cls])
        for pos, sample_idx in enumerate(class_indices):
            cid = eligible[pos % len(eligible)]
            partitions[cid].append(sample_idx)

    return partitions


def _quantity_skew_partition(indices: List[int], num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(int(seed) + 203)
    shuffled = list(indices)
    random.Random(int(seed) + 204).shuffle(shuffled)
    alpha = max(float(alpha), 1e-3)
    weights = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
    counts = np.floor(weights * len(shuffled)).astype(int)
    while counts.sum() < len(shuffled):
        counts[int(rng.integers(0, num_clients))] += 1
    while counts.sum() > len(shuffled):
        j = int(rng.integers(0, num_clients))
        if counts[j] > 0:
            counts[j] -= 1
    partitions = []
    cursor = 0
    for c in counts.tolist():
        take = shuffled[cursor: cursor + c]
        partitions.append(list(take))
        cursor += c
    return partitions


def _resolve_mp_context(num_workers: int, loader_mp_context: Optional[str]):
    if int(max(0, num_workers)) <= 0:
        return None
    if loader_mp_context is None:
        return None
    ctx_name = str(loader_mp_context).strip().lower()
    if ctx_name in {'', 'none', 'default'}:
        return None
    return torch.multiprocessing.get_context(ctx_name)


def _seed_worker(worker_id: int):
    # Keep worker-local RNGs deterministic while avoiding identical streams.
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def _loader_kwargs(
    batch_size: int,
    shuffle: bool,
    generator=None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    loader_mp_context: Optional[str] = None,
):
    num_workers = int(max(0, num_workers))
    kwargs: Dict[str, Any] = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': bool(pin_memory) if num_workers > 0 else False,
    }
    if generator is not None:
        kwargs['generator'] = generator
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(persistent_workers)
        kwargs['worker_init_fn'] = _seed_worker
        mp_ctx = _resolve_mp_context(num_workers=num_workers, loader_mp_context=loader_mp_context)
        if mp_ctx is not None:
            kwargs['multiprocessing_context'] = mp_ctx
    return kwargs


def _split_indices(total_size: int, seed: int, val_ratio: float, proxy_ratio: float, proxy_dataset_size: Optional[int]) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(total_size))
    rng = random.Random(int(seed) + 701)
    rng.shuffle(indices)
    if proxy_dataset_size is not None and int(proxy_dataset_size) > 0:
        proxy_n = min(int(proxy_dataset_size), total_size)
    else:
        proxy_n = int(round(total_size * max(0.0, float(proxy_ratio))))
    remaining = total_size - proxy_n
    val_n = min(remaining, int(round(total_size * max(0.0, float(val_ratio)))))
    proxy_idx = indices[:proxy_n]
    val_idx = indices[proxy_n: proxy_n + val_n]
    train_idx = indices[proxy_n + val_n:]
    if len(train_idx) <= 0:
        raise ValueError('Split produced empty train_idx; reduce proxy/val split.')
    return train_idx, proxy_idx, val_idx


def _build_partition(trainset, indices: List[int], num_clients: int, partition_scheme: str, seed: int, label_skew_classes: int, quantity_skew_alpha: float) -> List[List[int]]:
    targets = _extract_targets(trainset)
    scheme = str(partition_scheme).lower()
    rng = random.Random(int(seed) + 301)

    if scheme == 'iid':
        partitions = _iid_partition(indices, num_clients, rng)
    elif scheme == 'label-skew':
        partitions = _label_skew_partition(targets, indices, num_clients, label_skew_classes, int(seed))
    elif scheme == 'quantity-skew':
        partitions = _quantity_skew_partition(indices, num_clients, quantity_skew_alpha, int(seed))
    else:
        raise ValueError(f'Unsupported partition_scheme: {partition_scheme}')

    for i in range(num_clients):
        if len(partitions[i]) == 0:
            donor = max(range(num_clients), key=lambda x: len(partitions[x]))
            if len(partitions[donor]) <= 1:
                raise ValueError(f'Partition scheme {partition_scheme} produced an empty client with no donor margin.')
            partitions[i].append(partitions[donor].pop())

    return partitions


def _shutdown_dataloader(loader) -> None:
    if loader is None:
        return
    iterator = getattr(loader, '_iterator', None)
    if iterator is None:
        return
    shutdown = getattr(iterator, '_shutdown_workers', None)
    if callable(shutdown):
        try:
            shutdown()
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                "Failed to shut down DataLoader workers cleanly."
            ) from exc
    loader._iterator = None


def cleanup_dataloaders(dataloaders: Optional[Dict[str, Any]]) -> None:
    if not isinstance(dataloaders, dict):
        return
    for value in dataloaders.values():
        if isinstance(value, (list, tuple)):
            for item in value:
                _shutdown_dataloader(item)
        else:
            _shutdown_dataloader(value)


def _default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _load_dataset_pair(dataset_path: str, dataset_name: str):
    transform = _default_transform()
    name = str(dataset_name).lower()
    if name == 'cifar10':
        trainset = datasets.CIFAR10(
            root=dataset_path,
            train=True,
            download=False,
            transform=transform,
        )
        testset = datasets.CIFAR10(
            root=dataset_path,
            train=False,
            download=False,
            transform=transform,
        )
    elif name == 'cifar100':
        trainset = datasets.CIFAR100(
            root=dataset_path,
            train=True,
            download=False,
            transform=transform,
        )
        testset = datasets.CIFAR100(
            root=dataset_path,
            train=False,
            download=False,
            transform=transform,
        )
    elif name == 'femnist':
        trainset, testset = _load_femnist_datasets(dataset_path)
    else:
        raise ValueError(
            'Unsupported dataset_name. Use cifar10, cifar100, or femnist.'
        )
    return trainset, testset


def _proxy_version(
    *,
    dataset_name: str,
    proxy_indices: List[int],
    transform_identity: str,
) -> str:
    canonical = json.dumps(
        {
            "dataset": str(dataset_name).lower(),
            "proxy_indices": [int(index) for index in proxy_indices],
            "sample_order": "listed",
            "transform": str(transform_identity),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_federated_data_plan(
    *,
    dataset_path: str,
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    seed: int,
    partition_scheme: str,
    label_skew_classes: int,
    quantity_skew_alpha: float,
    val_ratio: float,
    proxy_ratio: float,
    proxy_dataset_size: Optional[int],
) -> FederatedDataPlan:
    """Partition once in the parent and return a spawn-serializable plan."""

    if int(num_clients) <= 0:
        raise ValueError("num_clients must be positive.")
    trainset, _ = _load_dataset_pair(dataset_path, dataset_name)
    train_idx, proxy_idx, val_idx = _split_indices(
        len(trainset),
        int(seed),
        float(val_ratio),
        float(proxy_ratio),
        proxy_dataset_size,
    )
    partitions = _build_partition(
        trainset=trainset,
        indices=train_idx,
        num_clients=int(num_clients),
        partition_scheme=str(partition_scheme),
        seed=int(seed),
        label_skew_classes=int(label_skew_classes),
        quantity_skew_alpha=float(quantity_skew_alpha),
    )
    transform_identity = "tensor-normalize-0.5-v1"
    return FederatedDataPlan(
        dataset_path=os.path.abspath(dataset_path),
        dataset_name=str(dataset_name).lower(),
        seed=int(seed),
        client_indices=tuple(
            tuple(int(index) for index in partition)
            for partition in partitions
        ),
        proxy_indices=tuple(int(index) for index in proxy_idx),
        validation_indices=tuple(int(index) for index in val_idx),
        partition_scheme=str(partition_scheme),
        label_skew_classes=int(label_skew_classes),
        quantity_skew_alpha=float(quantity_skew_alpha),
        batch_size=int(batch_size),
        transform_identity=transform_identity,
        proxy_version=_proxy_version(
            dataset_name=dataset_name,
            proxy_indices=proxy_idx,
            transform_identity=transform_identity,
        ),
    )


def build_server_dataloaders_from_plan(
    plan: FederatedDataPlan,
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    loader_mp_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Build only server-visible labeled proxy/validation/test loaders."""

    trainset, testset = _load_dataset_pair(
        plan.dataset_path,
        plan.dataset_name,
    )
    proxy_subset = Subset(trainset, list(plan.proxy_indices))
    validation_subset = (
        Subset(trainset, list(plan.validation_indices))
        if plan.validation_indices
        else None
    )
    kwargs = _loader_kwargs(
        batch_size=int(plan.batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
        loader_mp_context=loader_mp_context,
    )
    return {
        "proxy": DataLoader(proxy_subset, **kwargs),
        "val": (
            DataLoader(validation_subset, **kwargs)
            if validation_subset is not None
            else None
        ),
        "test": DataLoader(testset, **kwargs),
        "proxy_version": plan.proxy_version,
    }


def build_client_dataloaders_from_plan(
    plan: FederatedDataPlan,
    *,
    client_id: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    loader_mp_context: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build one private labeled loader and one input-only proxy loader."""

    client_id = int(client_id)
    if not 0 <= client_id < plan.num_clients:
        raise ValueError(f"Unknown client_id={client_id}.")
    trainset, _ = _load_dataset_pair(
        plan.dataset_path,
        plan.dataset_name,
    )
    private_subset = Subset(
        trainset,
        list(plan.client_indices[client_id]),
    )
    generator = torch.Generator()
    generator.manual_seed(int(plan.seed) + 1000 + client_id)
    private_loader = DataLoader(
        private_subset,
        **_loader_kwargs(
            batch_size=int(plan.batch_size),
            shuffle=True,
            generator=generator,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            persistent_workers=False,
            loader_mp_context=loader_mp_context,
        ),
    )
    proxy_loader = DataLoader(
        ProxyInputDataset(trainset, list(plan.proxy_indices)),
        **_loader_kwargs(
            batch_size=int(plan.batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
            persistent_workers=False,
            loader_mp_context=loader_mp_context,
        ),
    )
    return private_loader, proxy_loader


def get_dataloaders(
    dataset_path: str,
    dataset_name: str = 'cifar10',
    num_clients: int = 5,
    batch_size: int = 64,
    seed: Optional[int] = None,
    partition_scheme: str = 'iid',
    label_skew_classes: int = 2,
    quantity_skew_alpha: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_ratio: float = 0.1,
    proxy_ratio: float = 0.1,
    proxy_dataset_size: Optional[int] = None,
    persistent_workers: bool = False,
    loader_mp_context: Optional[str] = None,
    auxiliary_num_workers: int = 0,
):
    dataset_name = dataset_name.lower()
    trainset, testset = _load_dataset_pair(dataset_path, dataset_name)

    seed_i = int(seed) if seed is not None else 0
    torch.manual_seed(seed_i)
    np.random.seed(seed_i)
    random.seed(seed_i)

    train_idx, proxy_idx, val_idx = _split_indices(len(trainset), seed_i, val_ratio, proxy_ratio, proxy_dataset_size)
    partitions = _build_partition(
        trainset=trainset,
        indices=train_idx,
        num_clients=num_clients,
        partition_scheme=partition_scheme,
        seed=seed_i,
        label_skew_classes=label_skew_classes,
        quantity_skew_alpha=quantity_skew_alpha,
    )

    client_loaders = []
    partition_stats = []
    targets = _extract_targets(trainset)
    for i, part_indices in enumerate(partitions):
        part = Subset(trainset, part_indices)
        gen = torch.Generator()
        gen.manual_seed(seed_i + 1000 + i)
        client_loaders.append(DataLoader(
            part,
            **_loader_kwargs(
                batch_size=batch_size,
                shuffle=True,
                generator=gen,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                loader_mp_context=loader_mp_context,
            )
        ))

        subset_targets = [int(targets[idx]) for idx in part_indices]
        cls_hist = defaultdict(int)
        for y in subset_targets:
            cls_hist[y] += 1
        partition_stats.append({
            'client_id': i,
            'num_samples': len(part_indices),
            'num_classes': len(cls_hist),
            'max_class_frac': (max(cls_hist.values()) / max(1, len(part_indices))) if cls_hist else 0.0,
        })

    proxy_subset = Subset(trainset, proxy_idx)
    val_subset = Subset(trainset, val_idx) if len(val_idx) > 0 else None
    aux_workers = int(max(0, auxiliary_num_workers))
    proxy_loader = DataLoader(
        proxy_subset,
        **_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            generator=None,
            num_workers=aux_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            loader_mp_context=loader_mp_context,
        )
    )
    val_loader = DataLoader(
        val_subset,
        **_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            generator=None,
            num_workers=aux_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            loader_mp_context=loader_mp_context,
        )
    ) if val_subset is not None else None
    test_loader = DataLoader(
        testset,
        **_loader_kwargs(
            batch_size=batch_size,
            shuffle=False,
            generator=None,
            num_workers=aux_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
            loader_mp_context=loader_mp_context,
        )
    )

    return {
        'client': client_loaders,
        'proxy': proxy_loader,
        'val': val_loader,
        'test': test_loader,
        'partition_stats': partition_stats,
        'partition_scheme': str(partition_scheme),
        'split_sizes': {
            'train': len(train_idx),
            'proxy': len(proxy_idx),
            'val': len(val_idx),
            'test': len(testset),
        },
        'loader_config': {
            'client_num_workers': int(max(0, num_workers)),
            'auxiliary_num_workers': aux_workers,
            'pin_memory': bool(pin_memory),
            'persistent_workers': bool(persistent_workers),
            'loader_mp_context': loader_mp_context,
        },
    }
