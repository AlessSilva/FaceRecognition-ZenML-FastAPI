import pytest
from components.loader import TripletsDataLoadMethod, DataLoader, TradicionalDataLoadMethod
from components.exceptions import TripletsDataLoadError


@pytest.fixture
def set_up_dataset(tmp_path):
    class1_dir = tmp_path / "class1"
    class1_dir.mkdir(parents=True)
    (class1_dir / "img1.jpg").touch()
    (class1_dir / "img2.jpg").touch()

    class2_dir = tmp_path / "class2"
    class2_dir.mkdir(parents=True)
    (class2_dir / "img1.jpg").touch()
    (class2_dir / "img2.jpg").touch()

    return str(tmp_path)


@pytest.fixture
def set_up_dataset_one_image(tmp_path):
    class1_dir = tmp_path / "class1"
    class1_dir.mkdir(parents=True)
    (class1_dir / "img1.jpg").touch()

    class2_dir = tmp_path / "class2"
    class2_dir.mkdir(parents=True)
    (class2_dir / "img1.jpg").touch()


    return str(tmp_path)


def test_data_loader_triplets_correctly(set_up_dataset):
    data_loader_method = TripletsDataLoadMethod()
    data_loader = DataLoader(dataset_path=set_up_dataset,
                             data_load_method=data_loader_method)
    triplets = data_loader.load()
    assert len(triplets) > 0
    for triplet in triplets:
        assert len(triplet) == 3


def test_data_loader_tradicional_correctly(set_up_dataset):
    data_loader_method = TradicionalDataLoadMethod()
    data_loader = DataLoader(dataset_path=set_up_dataset,
                             data_load_method=data_loader_method)
    image_paths, labels = data_loader.load()
    assert len(image_paths) > 0
    assert len(labels) > 0
    assert len(labels) == len(image_paths) == 4


def test_data_loader_error():
    data_loader_method = TripletsDataLoadMethod()

    error_message = "TripletsDataLoadError: Failed to loading dataset for dataset 'invalid_path'"
    with pytest.raises(TripletsDataLoadError, match=error_message):
        data_loader = DataLoader(dataset_path='invalid_path',
                                 data_load_method=data_loader_method)
        triplets = data_loader.load()


def test_data_loader_insufficient_images(set_up_dataset_one_image):
    data_loader_method = TripletsDataLoadMethod()
    data_loader = DataLoader(dataset_path=set_up_dataset_one_image,
                             data_load_method=data_loader_method)
    triplets = data_loader.load()
    assert len(triplets) == 0
