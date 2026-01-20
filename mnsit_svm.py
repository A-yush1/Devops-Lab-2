import numpy as np
import struct
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import time

# ---------------------------
# Dataset directory
# ---------------------------
DATA_DIR = "/kaggle/input/mnist-dataset/"

# ---------------------------
# Resolve file (handles directory case)
# ---------------------------
def resolve_path(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        if len(files) == 0:
            raise FileNotFoundError(f"No files inside directory: {path}")
        return os.path.join(path, files[0])
    return path

def find_file(possible_names):
    for name in possible_names:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return resolve_path(path)
    raise FileNotFoundError(f"None found: {possible_names}")

# ---------------------------
# Locate MNIST files
# ---------------------------
train_images_path = find_file([
    "train-images-idx3-ubyte",
    "train-images.idx3-ubyte"
])

train_labels_path = find_file([
    "train-labels-idx1-ubyte",
    "train-labels.idx1-ubyte"
])

test_images_path = find_file([
    "t10k-images-idx3-ubyte",
    "t10k-images.idx3-ubyte"
])

test_labels_path = find_file([
    "t10k-labels-idx1-ubyte",
    "t10k-labels.idx1-ubyte"
])

print("Resolved paths:")
print(train_images_path)
print(train_labels_path)
print(test_images_path)
print(test_labels_path)

# ---------------------------
# Load IDX files
# ---------------------------
def load_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols)

def load_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

print("\nLoading MNIST data...")
X_train = load_images(train_images_path)
y_train = load_labels(train_labels_path)
X_test = load_images(test_images_path)
y_test = load_labels(test_labels_path)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------------------------
# Preprocessing
# ---------------------------
X_train = X_train / 255.0
X_test = X_test / 255.0

# ---------------------------
# Use subset (SVM is slow)
# ---------------------------
TRAIN_SAMPLES = 8000
TEST_SAMPLES = 2000

X_train = X_train[:TRAIN_SAMPLES]
y_train = y_train[:TRAIN_SAMPLES]
X_test = X_test[:TEST_SAMPLES]
y_test = y_test[:TEST_SAMPLES]

# ---------------------------
# Train SVM
# ---------------------------
svm = SVC(kernel="rbf", C=10, gamma="scale")

print("\nTraining SVM...")
start = time.time()
svm.fit(X_train, y_train)
print(f"Training time: {time.time() - start:.2f} seconds")

# ---------------------------
# Evaluation
# ---------------------------
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
