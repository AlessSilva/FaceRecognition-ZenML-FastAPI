from zenml import step
import os
import logging
import tensorflow as tf
from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer
from typing import Annotated
from components.database import setup_database, setup_faiss_index, add_vector_to_index


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def register_database(encoder: tf.keras.Model,
                      traditional_dataset: tf.data.Dataset,
                      ) -> Annotated[str, "database_path"]:
    try:
        logger.info(f"Register database step")
        database_path = "data.db"
        setup_database()
        index = setup_faiss_index(dimension=128)
        for batch in traditional_dataset:
            images, labels = batch[0], batch[1]
            embeddings = encoder(images, training=False)
            for embedding, label in zip(embeddings, labels):
                metadata = f"Label: {label.numpy()}"
                add_vector_to_index(index, embedding.numpy(), metadata)  
        return database_path

    except Exception as e:
        logger.info(f"Error register database step")
        raise e
