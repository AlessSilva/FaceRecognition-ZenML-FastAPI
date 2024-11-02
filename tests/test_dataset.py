import pytest
import tensorflow as tf
from components.dataset import TripletsDataset
from components.preprocess import PreprocessDecode, PreprocessResize
import os
from PIL import Image
import numpy as np


@pytest.fixture(autouse=True)
def verify_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if not physical_devices:
        print("No GPU found. Running on CPU only.")
    else:
        print(f"GPU detected: {physical_devices}")


@pytest.fixture
def example_triplet_paths(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    anchor_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    anchor_image.save(img_dir / "anchor.jpg")

    positive_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    positive_image.save(img_dir / "positive.jpg")

    negative_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    negative_image.save(img_dir / "negative.jpg")

    return [
        (str(img_dir / "anchor.jpg"), str(img_dir / "positive.jpg"), str(img_dir / "negative.jpg")),
        (str(img_dir / "anchor.jpg"), str(img_dir / "positive.jpg"), str(img_dir / "negative.jpg"))
    ]


def test_triplets_dataset_success(example_triplet_paths):
    preprocessors = [PreprocessDecode(), PreprocessResize()]

    params = {
        'channels': 3,
        'decode': 'jpeg',
        'target_size': (224, 224),
        'batch_size': 2
    }

    triplets_dataset = TripletsDataset()

    dataset = triplets_dataset.prepare(
        paths=example_triplet_paths,
        preprocessors=preprocessors,
        params=params
    )

    for batch in dataset:
        anchor, positive, negative = batch
        assert anchor.shape == (2, 224, 224, 3)
        assert positive.shape == (2, 224, 224, 3)
        assert negative.shape == (2, 224, 224, 3)
