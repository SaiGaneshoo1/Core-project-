import os
import numpy as np
from skimage import io, color, feature
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Set the path to your dataset
dataset_path = r'C:\Users\SAI GANESH\Documents\dataset\group1'

# Load images and extract simple features (average pixel values)
X = []
y = []

label_mapping = {}  # Map non-numeric labels to numeric values

def extract_features(image):
    # Replace this with your actual feature extraction method
    return np.mean(image, axis=(0, 1))

for idx, class_label in enumerate(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_label)
    
    if os.path.isdir(class_path):  # Ensure it's a directory
        label_mapping[class_label] = idx
        for file_name in os.listdir(class_path):
            image_path = os.path.join(class_path, file_name)
            
            if os.path.isfile(image_path):  # Ensure it's a file
                try:
                    # Load the image
                    image = io.imread(image_path)
                    
                    # Ensure the image has three channels (remove alpha channel if exists)
                    if image.shape[2] == 4:
                        image = image[:, :, :3]
                    
                    features = extract_features(image)
                    X.append(features)
                    y.append(label_mapping[class_label])
                except Exception as e:
                    print(f"Error loading image: {image_path}")
                    print(f"Error details: {e}")

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
clf = SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
