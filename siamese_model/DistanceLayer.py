import tensorflow as tf
import keras
from tensorflow.keras import layers


@keras.saving.register_keras_serializable()
class DistanceLayer(layers.Layer):
    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return (ap_distance, an_distance)
