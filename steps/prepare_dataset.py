from zenml import step
import os
import logging
import tensorflow as tf
from components.loader import DataLoader, TradicionalDataLoadMethod
from components.dataset import TraditionalDataset
from components.preprocess import PreprocessDecode, PreprocessResize
from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer
from typing import Annotated


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def prepare_dataset(dataset_path: str,
                    ) -> Annotated[tf.data.Dataset, "traditional_dataset"]:
    try:
        logger.info(f"Prepare traditional datatset step")
        loader = DataLoader(dataset_path, TradicionalDataLoadMethod())
        traditional = loader.load()
        preprocessors = [PreprocessDecode(), PreprocessResize()]
        params = {
            'channels': 3,
            'decode': 'jpeg',
            'target_size': (224, 224),
            'batch_size': 4
        }
        dataset = TraditionalDataset().prepare(paths=traditional,
                                               preprocessors=preprocessors,
                                               params=params,
                                               )
        return dataset

    except Exception as e:
        logger.info(f"Error prepare traditional step")
        raise e
