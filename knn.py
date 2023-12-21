import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set the path to the dataset
data_path = r'C:\Users\SAI GANESH\Documents\dataset\group1'

# Load images and labels from the dataset
image_list = []
labels = []
target_size = (224, 224)  # Set the desired size

for label, folder_name in enumerate(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)

            # Error-checking: Ensure the image is loaded successfully
            try:
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    raise Exception(f"Error loading image: {img_path}")

                # Resize the image to a consistent size
                img = cv2.resize(img, target_size)

                # Add a check for empty images before color conversion
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                image_list.append(img)
                labels.append(label)
            except Exception as e:
                print(f"Error processing image: {img_path}. {e}")

# Convert lists to NumPy arrays
X = np.array(image_list)
y = np.array(labels)

# Flatten the images to one-dimensional arrays
X = X.reshape(X.shape[0], -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
