from __future__ import annotations

from pathlib import Path

from PIL import Image
import numpy as np

from data_utils import (
    _dirichlet_label_partition,
    _load_dataset_pair,
)


def _write_rgb(path: Path, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size, size, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_classwise_dirichlet_partition_loses_no_samples():
    targets = [index % 10 for index in range(200)]
    indices = list(range(200))
    partitions = _dirichlet_label_partition(
        targets,
        indices,
        num_clients=20,
        alpha=0.5,
        seed=3,
    )
    flattened = [index for partition in partitions for index in partition]
    assert sorted(flattened) == indices
    assert len(flattened) == len(set(flattened))
    assert len(partitions) == 20


def test_cinic10_imagefolder_loader_requires_and_reads_ten_classes(tmp_path):
    for split in ("train", "test"):
        for class_id in range(10):
            _write_rgb(
                tmp_path / split / f"class_{class_id:02d}" / "sample.png",
                size=32,
            )
    train, test = _load_dataset_pair(str(tmp_path), "cinic10")
    assert len(train.classes) == 10
    assert len(train) == 10
    assert len(test) == 10
    image, label = train[0]
    assert tuple(image.shape) == (3, 32, 32)
    assert isinstance(label, int)


def test_tiny_imagenet_official_validation_annotations_are_parsed(tmp_path):
    class_names = [f"n{class_id:08d}" for class_id in range(200)]
    for class_name in class_names:
        _write_rgb(
            tmp_path / "train" / class_name / "images" / "sample.png",
            size=64,
        )
    val_image = "val_0.JPEG"
    _write_rgb(tmp_path / "val" / "images" / val_image, size=64)
    (tmp_path / "val" / "val_annotations.txt").write_text(
        f"{val_image}\t{class_names[0]}\t0\t0\t64\t64\n",
        encoding="utf-8",
    )
    train, test = _load_dataset_pair(str(tmp_path), "tiny-imagenet-200")
    assert len(train.classes) == 200
    assert len(train) == 200
    assert len(test) == 1
    image, label = test[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert label == train.class_to_idx[class_names[0]]
