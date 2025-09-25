import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        model_path = os.path.join("artifacts", "training", "model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = load_model(model_path)

    def predict(self):
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # normalize

        pred = self.model.predict(test_image)
        print("Raw model output:", pred, "Shape:", pred.shape)

        # Softmax handling
        pred_class = np.argmax(pred, axis=1)[0]
        prediction = 'Tumor' if pred_class == 1 else 'Normal'

        return [{"image": prediction}]

