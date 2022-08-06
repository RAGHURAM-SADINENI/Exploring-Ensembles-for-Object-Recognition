import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import train_models


class EnsembleModel(keras.Model):
    def __init__(self):
        super(EnsembleModel, self).__init__()
        self.deep_model = train_models.Model(mode="deep")
        self.deep_model.compile(
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=["accuracy"]
        )
        self.deep_model.load_weights("S:\Documents\LSU\Machine Learning\GroupProject\mnist_models\max_deep_5ep\\")

        self.medium_model = train_models.Model(mode="medium")
        self.medium_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )
        self.medium_model.load_weights("S:\Documents\LSU\Machine Learning\GroupProject\mnist_models\max_medium_5ep\\")

        self.shallow_model = train_models.Model(mode="shallow")
        self.shallow_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )
        self.shallow_model.load_weights("S:\Documents\LSU\Machine Learning\GroupProject\mnist_models\max_shallow_5ep\\")

        self.softmax = layers.Softmax()


    def call(self, x):
        x = self.deep_model(x) + self.medium_model(x) + self.shallow_model(x)

        return self.softmax(x)


(x_train, y_train), (x_test, y_test) = train_models.load_mnist()
batch_size = 64

model = EnsembleModel()

model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
)

model.evaluate(x_test, y_test, batch_size=batch_size)