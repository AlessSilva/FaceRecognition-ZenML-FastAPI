import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import logging
import zipfile
from components.exceptions import DatasetDownloadError
import kagglehub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestor(ABC):

    def __init__(self,
                 root_path: Optional[str] = None,
                 train_path: Optional[str] = None,
                 dataset_name: Optional[str] = None):
        self.root_path = root_path
        self.train_path = train_path
        self.dataset_name = dataset_name

    @abstractmethod
    def ingest(self,) -> str:
        pass


class KaggleDataIngestor(DataIngestor):

    def ingest(self,) -> Tuple[List[str], List[int], List[str]]:
        def download(dataset_name) -> str:
            try:
                logger.info(f"Downloading dataset {dataset_name} from kaggle")
                root_path = kagglehub.dataset_download(dataset_name)
                logger.info(f"Dataset downloaded to {root_path}")
                return root_path

            except Exception as e:
                logger.error(f"Error downloading dataset {dataset_name}: {e}")
                raise DatasetDownloadError(dataset_name, "Failed to download dataset")

        try:
            if self.dataset_name is None:
                raise ValueError("Dataset name is required")

            self.root_path = download(self.dataset_name)

            if self.train_path and os.path.exists(os.path.join(self.root_path, self.train_path)):
                logger.info(f"Train path {train_path} found in dataset")
                dataset_train_path = os.path.join(self.root_path,
                                                  self.train_path)
            else:
                dataset_train_path = self.root_path

            return dataset_train_path

        except Exception as e:
            logger.info(f"Error ingesting dataset: {e}")
            raise e


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(name: str,
                          root_path: Optional[str] = None,
                          train_path: Optional[str] = None,
                          dataset_name: Optional[str] = None) -> DataIngestor:
        if name == "kaggle":
            return KaggleDataIngestor(dataset_name=dataset_name,
                                      train_path=train_path)
        else:
            raise ValueError(f"Unsupported data ingestor: {name}")
