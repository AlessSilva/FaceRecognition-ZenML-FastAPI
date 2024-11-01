import pytest
from unittest.mock import patch
from components.siamese_model.functions import (create_siamese_model,
                                                create_embedding_model,
                                                load_encoder_and_siamese_network)
from components.exceptions import SiameseModelCreateError
import tensorflow as tf


@pytest.mark.skip
def test_create_embedding_model():
    model = create_embedding_model()
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 128)

@pytest.mark.skip
@patch("components.siamese_model.functions.tf.keras.applications.VGG16", side_effect=Exception("Model error"))
def test_create_embedding_model_error(mock_erro):
    with pytest.raises(SiameseModelCreateError) as exc_info:
        create_embedding_model()
    assert str(exc_info.value) == "SiameseModelCreateError: Error creating Encoder Model"


@pytest.mark.skip
@patch("tensorflow.keras.models.load_model")
def test_load_encoder_and_siamese_network(mock_load_model):
    mock_encoder = tf.keras.Sequential(name="Mock_Encoder")
    mock_siamese_network = tf.keras.Sequential(name="Mock_Siamese_Network")
    mock_load_model.side_effect = [mock_encoder, mock_siamese_network]
    
    encoder, siamese_network = load_encoder_and_siamese_network("encoder_path", "siamese_network_path")
    assert encoder.name == "Mock_Encoder", "The encoder model was not loaded correctly."
    assert siamese_network.name == "Mock_Siamese_Network", "The siamese network model was not loaded correctly."



@patch("tensorflow.keras.models.load_model", side_effect=OSError("File not found"))
def test_load_encoder_and_siamese_network_failure(mock_load_model):
    with pytest.raises(OSError):
        load_encoder_and_siamese_network("invalid_encoder_path", "invalid_siamese_network_path")


@pytest.mark.skip
def test_create_new_siamese_model():
    siamese_model = create_siamese_model()
    assert isinstance(siamese_model.siamese_network, tf.keras.Model)
    assert isinstance(siamese_model.encoder, tf.keras.Model)
    assert isinstance(siamese_model, tf.keras.Model)
    assert len(siamese_model.siamese_network.layers) > 0
    assert len(siamese_model.encoder.layers) > 0
