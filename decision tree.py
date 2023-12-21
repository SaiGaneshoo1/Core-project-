import os
import numpy as np
from skimage import io, color, feature
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Function to extract color histogram features from images
def extract_features(image):
    # Convert the image to HSV color space
    hsv_image = color.rgb2hsv(image)

    # Calculate histograms for each channel
    hist_h = np.histogram(hsv_image[:, :, 0], bins=50, range=(0, 1))[0]
    hist_s = np.histogram(hsv_image[:, :, 1], bins=50, range=(0, 1))[0]
    hist_v = np.histogram(hsv_image[:, :, 2], bins=50, range=(0, 1))[0]

    # Concatenate the histograms to create the feature vector
    features = np.concatenate([hist_h, hist_s, hist_v])
    
    return features

# Set the path to your dataset
dataset_path = r'C:\Users\SAI GANESH\Documents\dataset\group1'

# Define parameters
test_size = 0.2

# Load images and extract color histogram features
X = []
y = []

label_mapping = {}  # Map non-numeric labels to numeric values

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

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Decision Tree classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)
