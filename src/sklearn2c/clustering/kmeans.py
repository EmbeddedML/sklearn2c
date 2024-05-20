import joblib
from sklearn.cluster import KMeans
from .clus_exporter import kMeansExporter

class Kmeans:
    def __init__(self, **kwargs) -> None:
        self.clus = KMeans(**kwargs)

    def train(self, train_samples, save_path=None):
        self.clus.fit(train_samples, save_path)
        if save_path:
            joblib.dump(self, save_path)

    def predict(self, test_samples):
        result = self.clus.predict(test_samples)
        return result

    @staticmethod
    def load(filename: str) -> "Kmeans":
        with open(filename, "rb") as joblib_file:
            model = joblib.load(joblib_file)
        if not isinstance(model, Kmeans):
            raise TypeError(
                f"Expected an object of type Kmeans, but got {type(model)} instead."
            )
        return model

    def export(self, filename="kmeans_clus_config"):
        kMeansWriter = kMeansExporter(self.clus)
        kMeansWriter.export(filename)
