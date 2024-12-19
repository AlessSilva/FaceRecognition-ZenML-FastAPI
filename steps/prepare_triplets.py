from zenml import step
import os
import logging
import tensorflow as tf
from components.loader import DataLoader, TripletsDataLoadMethod
from components.dataset import TripletsDataset
from components.preprocess import PreprocessDecode, PreprocessResize
from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer
from typing import Annotated


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def prepare_triplets(dataset_path: str,
                    ) -> Annotated[tf.data.Dataset, "triplets_dataset"]:
    try:
        logger.info(f"Prepare triplets step")
        loader = DataLoader(dataset_path, TripletsDataLoadMethod())
        triplets = loader.load()
        logger.info(triplets[0])
        preprocessors = [PreprocessDecode(), PreprocessResize()]
        params = {
            'channels': 3,
            'decode': 'jpeg',
            'target_size': (224, 224),
            'batch_size': 4
        }
        dataset = TripletsDataset().prepare(paths=triplets,
                                            preprocessors=preprocessors,
                                            params=params,
                                            )
        return dataset

    except Exception as e:
        logger.info(f"Error prepare triplets step")
        raise e
