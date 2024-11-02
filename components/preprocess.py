from abc import ABC, abstractmethod
import tensorflow as tf
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Preprocess(ABC):
    @abstractmethod
    def apply(self,
              image: Optional[tf.Tensor] = None,
              params: Dict = {}) -> tf.Tensor:
        pass


class PreprocessDecode(Preprocess):
    def apply(self,
              image: Optional[tf.Tensor] = None,
              params: Dict = {}) -> tf.Tensor:
        try:
            channels = params.get('channels', 3)
            decode = params.get('decode', 'jpeg')
            dtype = params.get('dtype', tf.float32)
            if decode == 'jpeg':
                image = tf.image.decode_jpeg(image, channels=channels)
            elif decode == 'png':
                image = tf.image.decode_png(image, channels=channels)
            elif decode == 'bmp':
                image = tf.image.decode_bmp(image, channels=channels)
            else:
                raise ValueError(f"Unsupported decode format: {decode}")

            return tf.cast(image, dtype)

        except tf.errors.InvalidArgumentError as e:
            raise ValueError(f"Error decoding image: {e}")


class PreprocessResize(Preprocess):
    def apply(self,
              image: Optional[tf.Tensor] = None,
              params: Dict = {}) -> tf.Tensor:
        try:
            target_size = params.get('target_size', (224, 224))
            normalize = params.get('normalize', True)
            
            image = tf.image.resize(image, target_size)
            
            if normalize:
                image = image / 255.0
  
            return image

        except tf.errors.InvalidArgumentError as e:
            raise ValueError(f"Error resizing image: {e}")
