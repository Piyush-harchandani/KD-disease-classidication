import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self):
        """Prepare validation data generator."""
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Load a trained Keras model from disk."""
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """Load model, run evaluation, and save results."""
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """Save evaluation scores to a JSON file."""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log params & metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            try:
                if tracking_url_type_store != "file" and "dagshub" not in mlflow.get_tracking_uri():
                    # ✅ Local MLflow server with registry
                    mlflow.keras.log_model(
                        self.model,
                        artifact_path="model",
                        registered_model_name="VGG16Model"
                    )
                else:
                    # ✅ DagsHub or file store → save model manually as artifact
                    local_path = Path("saved_model")
                    mlflow.keras.save_model(self.model, path=local_path)
                    mlflow.log_artifacts(str(local_path), artifact_path="model")

            except Exception as e:
                print(f"⚠️ Model logging fallback due to error: {e}")
                local_path = Path("saved_model")
                mlflow.keras.save_model(self.model, path=local_path)
                mlflow.log_artifacts(str(local_path), artifact_path="model")