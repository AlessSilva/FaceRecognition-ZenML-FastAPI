import pytest
from unittest.mock import patch
from steps.ingest_dataset import ingest_dataset
from steps.prepare_triplets import prepare_triplets
from components.ingestor import KaggleDataIngestor


def test_ingest_step():

    with patch.object(KaggleDataIngestor, "ingest", return_value="path/downloaded/dataset"):
        result = ingest_dataset(dataset_name="path/to/data",
                                train_path="path/to/train")
        assert result == "path/downloaded/dataset"


def test_prepare_triplets():
    pass
