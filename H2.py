import pickle
import os
import pandas as pd
import numpy as np
train_file = "/kaggle/input/fii-nn-2025-homework-2/extended_mnist_train.pkl"
test_file = "/kaggle/input/fii-nn-2025-homework-2/extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten())
    train_labels.append(label)

test_data = []
for image, label in test:
    test_data.append(image.flatten())

X_train = np.array(train_data) / 255.0
y_train = np.array(train_labels)
m, n_features = X_train.shape
n_classes = 10

y_onehot = np.zeros((m, n_classes))
y_onehot[np.arange(m), y_train] = 1

np.random.seed(42)
W = np.random.randn(n_features, n_classes) * np.sqrt(1.0 / n_features)
b = np.zeros((1, n_classes))

epochs = 150
learning_rate = 0.4
decay = 0.1
batch_size = 128
target_accuracy = 0.95

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0) + 1e-8
X_train = (X_train - mean) / std

for epoch in range(epochs):
    lr = learning_rate / (1 + decay * epoch)
    indices = np.arange(m)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_onehot = y_onehot[indices]
    y_train = y_train[indices]

    for i in range(0, m, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_onehot[i:i+batch_size]

        logits = np.dot(X_batch, W) + b
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss = -np.mean(np.sum(y_batch * np.log(probs + 1e-8), axis=1))

        dlogits = (probs - y_batch) / batch_size
        dW = np.dot(X_batch.T, dlogits)
        db = np.sum(dlogits, axis=0, keepdims=True)

        W -= lr * dW
        b -= lr * db

    logits_full = np.dot(X_train, W) + b
    y_pred_train = np.argmax(logits_full, axis=1)
    acc = np.mean(y_pred_train == y_train)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    if acc >= target_accuracy:
        print(f"Oprire anticipată: acuratețea a atins pragul de {target_accuracy*100:.1f}% la epoca {epoch+1}.")
        break

X_test = np.array(test_data) / 255.0
X_test = (X_test - mean) / std
logits_test = np.dot(X_test, W) + b
predictions = np.argmax(logits_test, axis=1)

# This is how you prepare a submission for the competition
predictions_csv = {
    "ID": [],
    "target": [],
}

for i, label in enumerate(predictions):
    predictions_csv["ID"].append(i)
    predictions_csv["target"].append(label)

df = pd.DataFrame(predictions_csv)
df.to_csv("submission.csv", index=False)