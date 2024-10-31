import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
e

class CIFAR10(object):
    def __init__(self, train_images, train_labels):
        """
        Data is pre-loaded and passed to __init__ directly.
        Store the data as instance variables.
        """
        self.model = None
        self.train_images = train_images
        self.train_labels = train_labels
        

    def training(self):
        """
        Training process of the model. Store the model as an instance variable.
        """

        self.model = models.Sequential()

        # Input is a vector of size 32x32x3
        self.model.add(Input(shape=(32, 32, 3)))

        # CNN1 is 7x7x3x64 
        self.model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # CNN2 is 3x3x64x128
        self.model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # CNN31 is 3x3x128x256
        self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))

        # CNN32 is 3x3x256x256

        self.model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(layers.MaxPooling2D((2, 2)))

        # FNN4 is of a dimension 4096x10
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(4096, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

       
        self.model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        self.train_accuracies = []
        self.test_accuracies = []
        self.training_times = []
        self.train_errors = []
        self.test_errors = []
        
        
        epochs = 10
        for epoch in range(epochs):
            start = time.time()
            history = self.model.fit(self.train_images, self.train_labels, epochs=1, batch_size=64, validation_data=(self.train_images, self.train_labels), verbose=0)
            end = time.time()

            self.training_times.append(end - start)

            train_loss, train_accuracy = self.model.evaluate(self.train_images, self.train_labels, verbose=0)
            test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
            self.train_errors.append(1 - train_accuracy)
            self.test_errors.append(1 - test_accuracy)
            self.train_accuracies.append(train_accuracy)
            self.test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch+1}/{epochs} - Training Time: {self.training_times[-1]:.2f}s, Train Accuracy: {self.train_accuracies[-1]:.4f}, Test Accuracy: {self.test_accuracies[-1]:.4f}")
            

    def testing(self, test_images):
        """
        Use the trained model to predict the labels.
        Return an n*1 array that contains the label for each test image.
        """
        predictions = self.model.predict(test_images)
        argmax_predictions = np.argmax(predictions, axis=1)
        return argmax_predictions.reshape(-1, 1)

# Dataset preparation:
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()  # Load and split the dataset
train_images, test_images = train_images / 255.0, test_images / 255.0  # Normalize pixel values to be between 0 and 1

# Model training
model = CIFAR10(train_images, train_labels)
model.training()


# time per epoch graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), model.training_times, marker='o', linestyle='-', color='g')
plt.title('Training Time per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Training Time (seconds)')
plt.grid(True)
plt.show()

# training and testing errors graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), model.train_errors, marker='o', linestyle='-', color='b', label='Training Error')
plt.plot(range(1, 11), model.test_errors, marker='o', linestyle='-', color='r', label='Testing Error')
plt.title('Training and Testing Errors per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.show()

