from zenml import step
import kagglehub
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def download_dataset(dataset_name: str,
                     train_path: str | None
                     ) -> str:
    try:
        logger.info(f"Downloading dataset {dataset_name} from kaggle")
        folder_path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to {folder_path}")

        if train_path and os.path.exists(os.path.join(folder_path, train_path)):
            logger.info(f"Train path {train_path} found in dataset {dataset_name}")
            folder_path = os.path.join(folder_path, train_path)
            return folder_path
        else:
            logger.info(f"Train path {train_path} not found in dataset {dataset_name}")
            raise ValueError(f"Train path {train_path} not found in dataset {dataset_name}")

        return folder_path
    except Exception as e:
        logger.info(f"Error downloading dataset {dataset_name}: {e}")
        raise e
