import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import zipfile
from components.exceptions import TripletsDataLoadError
import kagglehub
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoadMethod(ABC):

    @abstractmethod
    def load(self, dataset_path):
        pass


class TripletsDataLoadMethod(DataLoadMethod):
    def load(self, dataset_path):
        try:
            logger.info(f"Loading images from folder {dataset_path}")
            image_paths = []
            labels = []
            class_names = sorted(os.listdir(dataset_path)[:2])
            logger.info(f"Class names: {class_names}")

            class_to_label = {
                class_name: index for index, class_name in enumerate(class_names)
            }

            for class_name in class_names:
                class_folder = os.path.join(dataset_path, class_name)
                for filename in os.listdir(class_folder):
                    image_paths.append(os.path.join(class_folder, filename))
                    labels.append(class_to_label[class_name])

            logger.info(f"Number of images: {len(image_paths)}")
            logger.info(f"Number of labels: {len(labels)}")

            triplets = []
            max_files_per_class = 10
            class_to_indices = {class_name: [] for class_name in class_names}

            for idx, label in enumerate(labels):
                class_name = class_names[label]
                class_to_indices[class_name].append(image_paths[idx])

            for class_name, images in class_to_indices.items():
                num_images = len(images)
                if num_images < 2:
                    continue

                for i in range(min(num_images, max_files_per_class) - 1):
                    for j in range(i + 1, min(num_images, max_files_per_class)):
                        anchor = images[i]
                        positive = images[j]

                        neg_class_name = class_name
                        while neg_class_name == class_name:
                            neg_class_name = random.choice(class_names)

                        negative = random.choice(class_to_indices[neg_class_name])

                        triplets.append((anchor, positive, negative))
            logger.info(f"Number of triplets: {len(triplets)}")
            random.shuffle(triplets)
            return triplets
        except Exception as e:
            logger.info(f"Error loading images: {e}")
            raise TripletsDataLoadError(dataset_path, f"Failed to loading dataset")

class DataLoader:
    def __init__(self,
                 dataset_path: str,
                 data_load_method: DataLoadMethod,):
        self.dataset_path = dataset_path
        self.data_load_method = data_load_method
    
    def load(self):
        return self.data_load_method.load(self.dataset_path)

