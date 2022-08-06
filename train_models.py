import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#             --- Variations ---
#
#  Deep     - all 3x3 kernels - 3x 32, 3x 64, 3x 128
#  Medium   - all 5x5 kernels - 2x 64, 2x 128, 2x 256
#  Shallow  - all 7x7 kernels - 1x 128, 1x 256, 1x 512
#
#   Next try all again with avg pool?
class Model(keras.Model):
    def __init__(self, pool='max', mode="shallow"):
        super(Model, self).__init__()

        if pool == 'max':
            self.pool1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
            self.pool2 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
            self.pool3 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")
        elif pool == 'avg':
            self.pool1 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")
            self.pool2 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")
            self.pool3 = layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding="same")

        # Convolutional and pooling layers.
        if mode is "shallow":
            self.conv_layers = [
                layers.Conv2D(128, kernel_size=7, padding="same", activation="relu"),
                self.pool1,
                layers.Conv2D(256, kernel_size=7, padding="same", activation="relu"),
                self.pool2,
                layers.Conv2D(512, kernel_size=7, padding="same", activation="relu"),
                self.pool3,
            ]
        elif mode is "medium":
            self.conv_layers = [
                layers.Conv2D(64, kernel_size=5, padding="same", activation="relu"),
                layers.Conv2D(64, kernel_size=5, padding="same", activation="relu"),
                self.pool1,
                layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),
                layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),
                self.pool2,
                layers.Conv2D(256, kernel_size=5, padding="same", activation="relu"),
                layers.Conv2D(256, kernel_size=5, padding="same", activation="relu"),
                self.pool3,
            ]
        elif mode is "deep":
            self.conv_layers = [
                layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
                self.pool1,
                layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
                self.pool2,
                layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
                layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
                self.pool3,
            ]

        self.global_pool =  layers.GlobalAveragePooling2D()
        self.decode_layer = layers.Dense(10, activation='softmax')


    def call(self, x):
        for l in self.conv_layers:
          x = l(x)

        # for l in self.dense_layers:
        #   x = l(x)

        x = self.global_pool(x)

        return self.decode_layer(x)


#   Load mnist, divide by 255 to convert values to be between zero and one.
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    (x_train, y_train), (x_test, y_test) = (tf.expand_dims(x_train.astype("float32"), -1) / 255, y_train.astype("float32")), (tf.expand_dims(x_test.astype("float32"), -1) / 255, y_test.astype("float32"))

    return (x_train, y_train), (x_test, y_test)


def load_cifar():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = (x_train.astype("float32") / 255, y_train.astype("float32")), (x_test.astype("float32") / 255, y_test.astype("float32"))

    return (x_train, y_train), (x_test, y_test)


def train_model(x_train, y_train, x_test, y_test):

    model = Model("max", "deep")

    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy"]
    )

    epochs = 5
    batch_size = 64

    # model.load_weights("S:\Documents\LSU\Machine Learning\GroupProject\mnist_models\max_shallow_5ep\\")

    for i in range(epochs):
        model.fit(x_train, y_train, batch_size=batch_size, epochs=1)
        model.save_weights(f"mnist_models\max_deep_{i+1}ep\\")

    model.evaluate(x_test, y_test, batch_size=batch_size)


# (x_train, y_train), (x_test, y_test) = load_mnist()
# train_model(x_train, y_train, x_test, y_test)