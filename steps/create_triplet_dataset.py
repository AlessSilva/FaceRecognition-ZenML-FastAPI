from zenml import step
from typing import Tuple, List
import tensorflow as tf
import logging
from zenml.integrations.tensorflow.materializers.tf_dataset_materializer import TensorflowDatasetMaterializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def create_triplet_dataset(triplets: List[Tuple[str, str, str]],
                           batch_size: int = 8
                           ) -> tf.data.Dataset:
    try:
        logger.info("Creating triplet TF dataset")

        def load_and_preprocess_image(image_path: str,
                                      target_size: Tuple[int, int] = (224, 224)
                                      ) -> tf.Tensor:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, target_size)
            image = image / 255.0
            return image

        def load_triplet(anchor: str,
                        positive: str,
                        negative: str
                        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
            anchor_image = load_and_preprocess_image(anchor)
            positive_image = load_and_preprocess_image(positive)
            negative_image = load_and_preprocess_image(negative)

            return (anchor_image, positive_image, negative_image)

        anchors, positives, negatives = zip(*triplets)
        anchors = tf.constant(anchors)
        positives = tf.constant(positives)
        negatives = tf.constant(negatives)

        dataset = tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))
        dataset = dataset.map(
            lambda anchor, positive, negative: load_triplet(anchor, positive, negative)
        )

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        logger.info(f"Dataset created with {batch_size} batch size")

        return dataset
    except Exception as e:
        logger.info(f"Error creating triplet dataset: {e}")
        raise e
