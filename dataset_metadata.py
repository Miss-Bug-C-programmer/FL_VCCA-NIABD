from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetNormalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]
    transform_identity: str
    raw_space_triggers: bool = False


_LEGACY_RGB_NORMALIZATION = DatasetNormalization(
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    transform_identity="tensor-normalize-0.5-v1",
)


DATASET_NORMALIZATIONS = {
    # Keep the established CIFAR paths byte-for-byte equivalent.
    "cifar10": _LEGACY_RGB_NORMALIZATION,
    "cifar100": _LEGACY_RGB_NORMALIZATION,
    "femnist": _LEGACY_RGB_NORMALIZATION,
    "tiny-imagenet-200": _LEGACY_RGB_NORMALIZATION,
    # Native one-channel MNIST tensors use the canonical train-set statistics.
    # Triggers are specified in raw [0, 1] pixel space and converted after
    # normalization, matching the CINIC-10 trigger semantics without changing
    # either existing dataset path.
    "mnist": DatasetNormalization(
        mean=(0.1307,),
        std=(0.3081,),
        transform_identity="tensor-normalize-mnist-official-v1",
        raw_space_triggers=True,
    ),
    # Official statistics shipped with the local CINIC-10 distribution.
    "cinic10": DatasetNormalization(
        mean=(0.47889522, 0.47227842, 0.43047404),
        std=(0.24205776, 0.23828046, 0.25874835),
        transform_identity="tensor-normalize-cinic10-official-v1",
        raw_space_triggers=True,
    ),
}


def canonical_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name).lower()
    if name in {"tiny_imagenet_200", "tinyimagenet200"}:
        return "tiny-imagenet-200"
    return name


def dataset_normalization(dataset_name: str) -> DatasetNormalization:
    name = canonical_dataset_name(dataset_name)
    try:
        return DATASET_NORMALIZATIONS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_NORMALIZATIONS))
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}; expected one of: "
            f"{supported}"
        ) from exc
