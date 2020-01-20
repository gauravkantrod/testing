import tensorflow.keras as keras
from kerastuner import RandomSearch


import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(28, 28, 1)
    ))

    model.add(keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ))
    model.add(keras.layers.Dense(10, activation="softmax")
              )

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )

    return model


tuner_search = RandomSearch(build_model, objective='val_accuracy', max_trials=3, directory='output',
                            project_name="Fashion_MNIST-Using-Keras_Tuner")

tuner_search.search(train_images, train_labels, epochs=3, validation_split=0.3)

best_model = tuner_search.get_best_models(num_models=1)[0]

best_model.summary()

best_model.fit(train_images, train_labels, epochs=5, validation_split=0.3, initial_epoch=3)
