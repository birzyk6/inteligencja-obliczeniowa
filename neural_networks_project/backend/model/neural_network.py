import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report


class NeuralNetworkVisualizer:
    def __init__(self):
        self.model = None
        self.layer_model = None
        self.training_history = None
        self.load_or_create_model()

    def create_model(self):
        """Create an improved CNN model for MNIST digit recognition with better digit 9 recognition"""
        model = keras.Sequential(
            [
                layers.Reshape((28, 28, 1), input_shape=(28, 28)),
                # First convolutional block with batch normalization
                layers.Conv2D(
                    32, (3, 3), activation="relu", padding="same", name="conv1"
                ),
                layers.BatchNormalization(name="bn1"),
                layers.Conv2D(
                    32, (3, 3), activation="relu", padding="same", name="conv1_2"
                ),
                layers.MaxPooling2D((2, 2), name="pool1"),
                layers.Dropout(0.25, name="dropout1"),
                # Second convolutional block
                layers.Conv2D(
                    64, (3, 3), activation="relu", padding="same", name="conv2"
                ),
                layers.BatchNormalization(name="bn2"),
                layers.Conv2D(
                    64, (3, 3), activation="relu", padding="same", name="conv2_2"
                ),
                layers.MaxPooling2D((2, 2), name="pool2"),
                layers.Dropout(0.25, name="dropout2"),
                # Third convolutional block
                layers.Conv2D(
                    128, (3, 3), activation="relu", padding="same", name="conv3"
                ),
                layers.BatchNormalization(name="bn3"),
                layers.Conv2D(
                    128, (3, 3), activation="relu", padding="same", name="conv3_2"
                ),
                layers.Dropout(0.25, name="dropout3"),
                # Fully connected layers
                layers.Flatten(name="flatten"),
                layers.Dense(512, activation="relu", name="dense1"),
                layers.BatchNormalization(name="bn4"),
                layers.Dropout(0.5, name="dropout4"),
                layers.Dense(256, activation="relu", name="dense2"),
                layers.Dropout(0.5, name="dropout5"),
                layers.Dense(10, activation="softmax", name="output"),
            ]
        )

        # Use a lower learning rate for better convergence
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train_model(self):
        """Train the model on MNIST dataset with data augmentation for better digit recognition"""
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = (
            keras.datasets.mnist.load_data()
        )  # Normalize pixel values and reshape for data augmentation
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # Reshape data to add channel dimension (28, 28) -> (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Data augmentation to improve generalization, especially for rotated digits like 6/9
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rotation_range=10,  # Small rotations to help distinguish 6 from 9
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            fill_mode="nearest",
        )

        # Learning rate scheduling for better convergence
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=3, min_lr=0.0001, verbose=1
        )

        # Early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7, restore_best_weights=True, verbose=1
        )

        print("Training model with data augmentation...")
        history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            steps_per_epoch=len(x_train) // 128,
            epochs=25,  # More epochs with early stopping
            validation_data=(x_test, y_test),
            callbacks=[lr_scheduler, early_stopping],
            verbose=1,
        )

        self.training_history = history.history  # Save the trained model
        self.model.save("model/trained_model.h5")

        # Save training history
        import json
        import numpy as np

        # Convert numpy float32 values to regular Python floats for JSON serialization
        def convert_float32(obj):
            if isinstance(obj, dict):
                return {key: convert_float32(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_float32(item) for item in obj]
            elif isinstance(obj, np.float32):
                return float(obj)
            else:
                return obj

        serializable_history = convert_float32(self.training_history)

        with open("model/training_history.json", "w") as f:
            json.dump(serializable_history, f)

        return history

    def load_or_create_model(self):
        """Load existing model or create and train a new one"""
        model_path = "model/trained_model.h5"
        history_path = "model/training_history.json"

        if os.path.exists(model_path):
            print("Loading existing model...")
            self.model = keras.models.load_model(model_path)

            if os.path.exists(history_path):
                import json

                with open(history_path, "r") as f:
                    self.training_history = json.load(f)
        else:
            print("Creating and training new model...")
            os.makedirs("model", exist_ok=True)
            self.model = self.create_model()
            self.train_model()

        # Create layer model for visualization
        self.create_layer_model()

    def create_layer_model(self):
        """Create a model that outputs all intermediate layer activations"""
        layer_outputs = []
        for layer in self.model.layers:
            if len(layer.output_shape) > 1:  # Skip input layer
                layer_outputs.append(layer.output)

        self.layer_model = keras.Model(inputs=self.model.input, outputs=layer_outputs)

    def predict(self, image):
        """Make prediction on a single image"""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return predicted_class, confidence

    def get_probabilities(self, image):
        """Get probability distribution for all classes"""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        predictions = self.model.predict(image, verbose=0)
        return predictions[0]

    def get_layer_outputs(self, image):
        """Get outputs from all layers for visualization"""
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        layer_outputs = self.layer_model.predict(image, verbose=0)
        return layer_outputs

    def get_model_info(self):
        """Get information about the model architecture"""
        model_info = {
            "total_params": self.model.count_params(),
            "layers": [],
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
        }

        for i, layer in enumerate(self.model.layers):
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": layer.output_shape,
                "trainable_params": layer.count_params(),
            }

            # Add specific info for different layer types
            if hasattr(layer, "filters"):
                layer_info["filters"] = layer.filters
            if hasattr(layer, "kernel_size"):
                layer_info["kernel_size"] = layer.kernel_size
            if hasattr(layer, "units"):
                layer_info["units"] = layer.units

            model_info["layers"].append(layer_info)

        return model_info

    def get_training_history(self):
        """Get training history information"""
        if self.training_history:
            return {
                "epochs": len(self.training_history["accuracy"]),
                "final_accuracy": self.training_history["accuracy"][-1],
                "final_val_accuracy": self.training_history["val_accuracy"][-1],
                "final_loss": self.training_history["loss"][-1],
                "final_val_loss": self.training_history["val_loss"][-1],
                "history": self.training_history,
            }
        else:
            return {"message": "No training history available"}

    def analyze_digit_performance(self):
        """Analyze the model's performance on each digit, especially digit 9"""
        print("Loading MNIST test dataset for analysis...")
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Normalize pixel values
        x_test = x_test.astype("float32") / 255.0

        # Make predictions
        predictions = self.model.predict(x_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_test, predicted_classes)

        # Calculate per-digit accuracy
        digit_accuracies = {}
        for digit in range(10):
            digit_mask = y_test == digit
            digit_predictions = predicted_classes[digit_mask]
            digit_accuracy = np.mean(digit_predictions == digit)
            digit_accuracies[digit] = digit_accuracy
            print(f"Digit {digit} accuracy: {digit_accuracy:.4f}")

        # Special focus on digit 9 confusion
        digit_9_mask = y_test == 9
        digit_9_predictions = predicted_classes[digit_9_mask]

        print(f"\nDigit 9 Analysis:")
        print(f"Total digit 9 samples: {np.sum(digit_9_mask)}")
        print(f"Correctly classified as 9: {np.sum(digit_9_predictions == 9)}")
        print(f"Digit 9 accuracy: {digit_accuracies[9]:.4f}")

        # What digits are confused with 9?
        misclassified_9 = digit_9_predictions[digit_9_predictions != 9]
        if len(misclassified_9) > 0:
            unique, counts = np.unique(misclassified_9, return_counts=True)
            print("Digit 9 confused with:")
            for digit, count in zip(unique, counts):
                print(
                    f"  Digit {digit}: {count} times ({count/len(digit_9_predictions)*100:.1f}%)"
                )

        return {
            "digit_accuracies": digit_accuracies,
            "confusion_matrix": cm.tolist(),
            "total_accuracy": np.mean(predicted_classes == y_test),
        }
