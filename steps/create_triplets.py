from zenml import step
from typing import Tuple, List
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@step
def create_triplets(image_paths: List[str],
                    labels: List[int],
                    class_names: List[str],
                    max_files_per_class: int = 10
                    ) -> List[Tuple[int, int, int]]:
    try:
        logger.info(f"Creating triplets")
        triplets = []
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
        logger.info(f"Error creating triplets: {e}")
        raise e
