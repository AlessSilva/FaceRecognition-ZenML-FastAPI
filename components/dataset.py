from abc import ABC, abstractmethod
import tensorflow as tf
from typing import List, Dict, Optional, Tuple
from components.preprocess import (Preprocess,
                                   PreprocessDecode,
                                   PreprocessResize)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dataset(ABC):
    @abstractmethod
    def prepare(self,
                paths,
                preprocessors: List[Preprocess],
                params: Dict) -> tf.data.Dataset:
        pass


class TripletsDataset(Dataset):
    def prepare(self,
                paths: List,
                preprocessors: List[Preprocess],
                params: Dict) -> tf.data.Dataset:

        try:
            logger.info("Creating triplet TF dataset")
            triplets = paths

            def apply_preprocess(image_path: str,) -> tf.Tensor:
                image = tf.io.read_file(image_path)
                for preprocessor in preprocessors:
                    image = preprocessor.apply(image, params)
                return image

            def load_triplet(anchor: str,
                             positive: str,
                             negative: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                anchor_image = apply_preprocess(anchor)
                positive_image = apply_preprocess(positive)
                negative_image = apply_preprocess(negative)

                return (anchor_image, positive_image, negative_image)

            anchors, positives, negatives = zip(*triplets)
            anchors = tf.constant(anchors)
            positives = tf.constant(positives)
            negatives = tf.constant(negatives)

            dataset = tf.data.Dataset.from_tensor_slices((anchors, positives, negatives))
            dataset = dataset.map(
                lambda anchor, positive, negative: load_triplet(anchor, positive, negative)
            )
            batch_size = params.get('batch_size', 8)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            logger.info(f"Dataset created with {batch_size} batch size")

            return dataset
        except Exception as e:
            logger.info(f"Error creating triplet dataset: {e}")
            raise e


class TraditionalDataset(Dataset):
    def prepare(self,
                paths: List,
                preprocessors: List[Preprocess],
                params: Dict) -> tf.data.Dataset:

        try:
            logger.info("Creating traditional TF dataset")
            image_paths, labels = paths[0], paths[1]

            def apply_preprocess(image_path: str,) -> tf.Tensor:
                image = tf.io.read_file(image_path)
                for preprocessor in preprocessors:
                    image = preprocessor.apply(image, params)
                return image

            def process_sample(image_path: str, label: str) -> Tuple[tf.Tensor, tf.Tensor]:
                image = apply_preprocess(image_path)
                return (image, label)

            image_paths = tf.constant(image_paths)
            labels = tf.constant(labels)
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            dataset = dataset.map(
                lambda image_path, label: process_sample(image_path, label)
            )
            batch_size = params.get('batch_size', 8)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            logger.info(f"Dataset created with {batch_size} batch size")

            return dataset
        except Exception as e:
            logger.info(f"Error creating triplet dataset: {e}")
            raise e
