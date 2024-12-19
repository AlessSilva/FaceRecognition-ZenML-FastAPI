import pytest
import tensorflow as tf
from components.dataset import TripletsDataset, TraditionalDataset
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


@pytest.fixture
def example_traditional_paths(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    image1 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image1.save(img_dir / "image1.jpg")

    image2 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image2.save(img_dir / "image2.jpg")

    image3 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image3.save(img_dir / "image3.jpg")

    image4 = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    image4.save(img_dir / "image4.jpg")

    return [
        [str(img_dir / "image1.jpg"), str(img_dir / "image2.jpg"), str(img_dir / "image3.jpg"), str(img_dir / "image4.jpg")],
        ["label1", "label2", "label1", "label1"]
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


def test_traditional_dataset_success(example_traditional_paths):
    preprocessors = [PreprocessDecode(), PreprocessResize()]

    params = {
        'channels': 3,
        'decode': 'jpeg',
        'target_size': (224, 224),
        'batch_size': 2
    }

    traditional_dataset = TraditionalDataset()

    dataset = traditional_dataset.prepare(
        paths=example_traditional_paths,
        preprocessors=preprocessors,
        params=params
    )

    for batch in dataset:
        images, labels = batch
        assert images.shape == (2, 224, 224, 3)
        assert labels.shape == (2,)
