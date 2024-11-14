from zenml import pipeline
from steps.ingest_dataset import ingest_dataset
from steps.prepare_triplets import prepare_triplets
from steps.training import training

@pipeline(enable_cache=False)
def train_pipeline(
    dataset_name: str = "wutheringwang/dog-face-recognition",
    train_path: str = "train",
    save_path: str = "model",
    epochs: int = 2,
):
    dataset_path = ingest_dataset(dataset_name=dataset_name, train_path=train_path)
    print(dataset_path)
    triplet_dataset = prepare_triplets(dataset_path)
    encoder, siamese_network = training(triplet_dataset, save_path=save_path, epochs=epochs)

if __name__ == "__main__":
    run = train_pipeline()
