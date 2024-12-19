import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras import metrics


@keras.saving.register_keras_serializable()
class SiameseModel(Model):

    def __init__(self, encoder, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()

        self.margin = margin
        self.siamese_network = siamese_network
        self.encoder = encoder
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        anchor_input, positive_input, negative_input = data
        with tf.GradientTape() as tape:
          loss = self._compute_loss(anchor_input, positive_input, negative_input)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, anchor_input, positive_input, negative_input):

        ap_distance, an_distance = self.siamese_network([anchor_input, positive_input, negative_input])
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)

        return tf.reduce_mean(loss)

    @property
    def metrics(self):
        return [self.loss_tracker]
