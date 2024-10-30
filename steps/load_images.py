from zenml import step
import os
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def load_images(folder_path: str
                ) -> Tuple[List[str], List[int], List[str]]:
    try:
        logger.info(f"Loading images from folder {folder_path}")
        image_paths = []
        labels = []
        class_names = sorted(os.listdir(folder_path)[:2])
        logger.info(f"Class names: {class_names}")

        class_to_label = {
            class_name: index for index, class_name in enumerate(class_names)
        }
        aux = 0
        for class_name in class_names:
            class_folder = os.path.join(folder_path, class_name)
            for filename in os.listdir(class_folder):
            image_paths.append(os.path.join(class_folder, filename))
            labels.append(class_to_label[class_name])

        logger.info(f"Number of images: {len(image_paths)}")
        logger.info(f"Number of labels: {len(labels)}")
        return image_paths, labels, class_names
    except Exception as e:
        logger.info(f"Error loading images: {e}")
        raise e
