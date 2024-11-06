import os
from zenml import step
import logging
import tensorflow as tf
import mlflow
from components.siamese_model.functions import create_siamese_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step(enable_cache=False)
def training(
    triplet_dataset: tf.data.Dataset,
    save_path: str,
    epochs: int = 10,
) -> tf.keras.Model:
    try:
        siamese_model = create_siamese_model()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"Diretório {save_path} criado!")
        with mlflow.start_run():
            logger.info("Starting training process...")
            for epoch in range(epochs):
                logger.info(f"\nEpoch {epoch + 1}/{epochs}")
                loss_per_epoch = []

                for batch in triplet_dataset:
                    loss = siamese_model.train_step(batch)
                    loss_value = loss['loss'].numpy()
                    loss_per_epoch.append(loss_value)

                avg_loss = sum(loss_per_epoch) / len(loss_per_epoch)
                logger.info(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

                mlflow.log_metric("average_loss", avg_loss, step=epoch)

                encoder_path = os.path.join(save_path, f"encoder_epoch_{epoch + 1}.keras")
                siamese_network_path = os.path.join(save_path, f"siamese_network_epoch_{epoch + 1}.keras")

                siamese_model.encoder.save(encoder_path)
                siamese_model.siamese_network.save(siamese_network_path)

                mlflow.keras.log_model(siamese_model.encoder, "encoder")
                mlflow.keras.log_model(siamese_model.siamese_network, "siamese_network")
    except Exception as e:
        logger.error("Error during training step", exc_info=e)
        raise e
    finally:
        mlflow.end_run()
