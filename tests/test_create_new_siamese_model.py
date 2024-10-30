import pytest
from siamese_model.functions import create_siamese_model
import tensorflow as tf

def test_create_new_siamese_model():
    siamese_model = create_siamese_model()
    assert isinstance(siamese_model.siamese_network, tf.keras.Model)
    assert isinstance(siamese_model.encoder, tf.keras.Model)
    assert isinstance(siamese_model, tf.keras.Model)
    assert len(siamese_model.siamese_network.layers) > 0
    assert len(siamese_model.encoder.layers) > 0
