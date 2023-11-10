import numpy as np
import pickle
from keras import layers, models

MODEL_PATH = 'model.keras'
EPOCH_COUNT = 100

def load_cifar10_data():
    train_data = {}
    for i in range(1, 6):
        with open(f'cifar-10-batches-py/data_batch_{i}', 'rb') as f:
            new_data = pickle.load(f, encoding='bytes')
            train_data = dict(list(train_data.items()) + list(new_data.items()))
    
    X_train = np.array(train_data[b'data'], dtype=np.float32) / 255.0
    y_train = np.array(train_data[b'labels'])
    
    with open(f'cifar-10-batches-py/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    
    # Preprocess data
    X_test = np.array(test_data[b'data'], dtype=np.float32) / 255.0
    y_test = np.array(test_data[b'labels'])

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = load_cifar10_data()

try:
    model = models.load_model(MODEL_PATH)
except:
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(256, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=EPOCH_COUNT)

# Оценка качества модели
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", test_accuracy)
model.save(MODEL_PATH)