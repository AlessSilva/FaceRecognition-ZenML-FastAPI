from zenml import step
import os
import logging
from components.ingestor import DataIngestorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def ingest_dataset(dataset_name: str,
                 train_path: str | None
                ) -> str:
    try:
        logger.info(f"Ingest dataset step")
        ingestor = DataIngestorFactory.get_data_ingestor(name="kaggle",
                                                         dataset_name=dataset_name,
                                                         train_path= train_path,)
        dataset_path = ingestor.ingest()
        return dataset_path

    except Exception as e:
        logger.info(f"Error ingest dataset step")
        raise e
