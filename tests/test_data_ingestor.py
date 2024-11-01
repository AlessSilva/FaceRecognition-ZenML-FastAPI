import pytest
from unittest.mock import patch 
from components.ingestor import KaggleDataIngestor
from components.exceptions import DatasetDownloadError

@pytest.fixture
def kaggle_data_ingestor():
    return KaggleDataIngestor(dataset_name="sample-dataset")


@patch("components.ingestor.kagglehub.dataset_download")
def test_kaggle_data_ingestor_download(mock_download, kaggle_data_ingestor):
    mock_download.return_value = "/path/to/dataset"
    result = kaggle_data_ingestor.ingest()
    assert result == "/path/to/dataset"
    mock_download.assert_called_once_with("sample-dataset")


@patch("components.ingestor.kagglehub.dataset_download", side_effect=Exception("Download error"))
def test_kaggle_data_ingestor_download_error(mock_download, kaggle_data_ingestor):
    with pytest.raises(DatasetDownloadError):
        kaggle_data_ingestor.ingest()


def test_kaggle_data_ingestor_dataset_not_defined(kaggle_data_ingestor):
    with pytest.raises(ValueError) as exc_info:
        kaggle_data_ingestor.dataset_name = None
        kaggle_data_ingestor.ingest()
    assert str(exc_info.value) == "Dataset name is required"
