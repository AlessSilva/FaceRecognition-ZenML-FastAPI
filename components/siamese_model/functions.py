import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from components.siamese_model.DistanceLayer import DistanceLayer
from components.siamese_model.SiameseModel import SiameseModel
from components.exceptions import SiameseModelCreateError
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_embedding_model():
    logger.info(f"Creating Encoder/embedding Model")
    try:
      base_model = tf.keras.applications.VGG16(include_top=False, input_shape=(224, 224, 3))
      num_layers_to_freeze = len(base_model.layers) - 7
      for i in range(num_layers_to_freeze):
        base_model.layers[i].trainable = False

      x = base_model.output
      x = layers.GlobalAveragePooling2D()(x)
      x = layers.Dense(128, activation='relu')(x)
      x = layers.Dense(128)(x)

      return Model(inputs=base_model.input, outputs=x)
    except Exception as e:
      logger.info(f"Error creating Encoder Model {e}")
      raise SiameseModelCreateError('Error creating Encoder Model')


def create_encoder_and_siamese_network(
  input_shape: Tuple[int, int, int] = (224, 224, 3)
  ):
    logger.info(f"Creating Encoder/Embedding Model and Siamese Network")
    encoder = create_embedding_model()
    try:
      anchor_input = layers.Input(shape=input_shape, name="Anchor_Input")
      positive_input = layers.Input(shape=input_shape, name="Positive_Input")
      negative_input = layers.Input(shape=input_shape, name="Negative_Input")

      encoded_a = encoder(anchor_input)
      encoded_p = encoder(positive_input)
      encoded_n = encoder(negative_input)

      distances = DistanceLayer()(
          encoded_a,
          encoded_p,
          encoded_n
      )

      siamese_network = Model(
          inputs=[anchor_input, positive_input, negative_input],
          outputs=distances,
          name="Siamese_Network"
      )
      return encoder, siamese_network
    except Exception as e:
      logger.info(f"Error creating Siamese Network {e}")
      raise SiameseModelCreateError("Error creating Siamese Network")


def load_encoder_and_siamese_network(
  encoder_path: str,
  siamese_network_path: str,
  ):
    logger.info(f"Loading Encoder/Embedding Model and Siamese Network")
    try:
      encoder = tf.keras.models.load_model(encoder_path)
      siamese_network = tf.keras.models.load_model(siamese_network_path)
      return encoder, siamese_network
    except Exception as e:
      logger.info(f"Error loading Encoder Model and Siamese Network {e}")
      raise e


def create_siamese_model(
  encoder_path: str | None = None,
  siamese_network_path: str | None = None,
  ):
    logger.info(f"Creating Siamese Model")
    try:
      if encoder_path is None or siamese_network_path is None:
        encoder, siamese_network = create_encoder_and_siamese_network()
      else:
        encoder, siamese_network = load_encoder_and_siamese_network(
          encoder_path,
          siamese_network_path
        )
      siamese_model = SiameseModel(encoder, siamese_network)
      siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, epsilon=1e-01))
      return siamese_model
    except Exception as e:
      logger.info(f"Error creating Siamese Model {e}")
      raise e
