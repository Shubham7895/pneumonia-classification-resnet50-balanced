import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pneumonia_utils

# Load test data
data = np.load("pneumoniamnist.npz")
test_images, test_labels = data['test_images'], data['test_labels']

# Preprocessing
test_images = pneumonia_utils.preprocess(test_images)

# Load model
model = tf.keras.models.load_model("resnet50_pneumonia.h5")
print("Model loaded.")

# Evaluate
predictions = (model.predict(test_images) > 0.5).astype("int32")

print("Classification Report:\n", classification_report(test_labels, predictions))
print("Confusion Matrix:\n", confusion_matrix(test_labels, predictions))
