class DatasetDownloadError(Exception):
    def __init__(self, dataset_name: str, message: str):
        self.dataset_name = dataset_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"DatasetDownloadError: {self.message} for dataset '{self.dataset_name}'"


class SiameseModelCreateError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"SiameseModelCreateError: {self.message}"


class TripletsDataLoadError(Exception):
    def __init__(self, dataset_name: str, message: str):
        self.dataset_name = dataset_name
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"TripletsDataLoadError: {self.message} for dataset '{self.dataset_name}'"
