import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('waste_classifier_mobilenet.h5')

# Path to the test images
test_dir = "test_dataset/"

# Set up image preprocessing and batch generation
datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Update this size based on your model input size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Make predictions
predictions = model.predict(test_generator, verbose=1)

# Get predicted class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get the actual labels from the generator
actual_labels = test_generator.classes

# Get the class labels from the generator
class_labels = test_generator.class_indices
class_names = {v: k for k, v in class_labels.items()}

# Count the number of images in each class
actual_class_counts = {class_name: 0 for class_name in class_names.values()}
predicted_class_counts = {class_name: 0 for class_name in class_names.values()}

for i in range(len(actual_labels)):
    actual_class = class_names[actual_labels[i]]
    predicted_class = class_names[predicted_labels[i]]
    actual_class_counts[actual_class] += 1
    predicted_class_counts[predicted_class] += 1

# Display the count for each class
print("Actual Class-wise image counts:")
for class_name, count in actual_class_counts.items():
    print(f"{class_name}: {count} images")

print("\nPredicted Class-wise image counts:")
for class_name, count in predicted_class_counts.items():
    print(f"{class_name}: {count} images")

# Generate classification report
print("\nClassification Report:")
report = classification_report(actual_labels, predicted_labels, target_names=class_names.values())
print(report)
