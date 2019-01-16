
# =====================================================================
#  CNN.py
# =====================================================================
#
#  Author:         (c) 2019 Antonio Pio Ricciardi & Matteo Prata
#  Created:        January  02, 2019

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from CNN_Preprocessing import *


class CNN:

    model = None

    def __init__(self, model=None):
        """
        The Convolutiona Neural Network to mount.

        :param model: the model to import in this CNN, None by default, in this case it creates a new one.
        """
        self.model = self.neural_model() if not model else model

    @staticmethod
    def neural_model():
        """
        NVIDIA model:

        (1) Image normalization to avoid saturation and make gradients work better.
        (2) Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
        (3) Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
        (4) Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
        (5) Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        (6) Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
        (.) Drop out (0.5)
        (7) Fully connected: neurons: 100, activation: ELU
        (8) Fully connected: neurons: 50, activation: ELU
        (9) Fully connected: neurons: 10, activation: ELU
        (.) Fully connected: neurons: 1 (output)

        :return: the Keras model
        """

        INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
        model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Conv2D(64, 3, 3, activation='elu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))
        model.summary()

        return model

    def train_model(self, train_dataset, test_dataset):
        """
        Trains this model in a supervised fashion and evaluates it.

        :param train_dataset: the traing dataset
        :param test_dataset: the test dataset
        """

        checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')

        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

        self.model.fit_generator(generator=generate_train_batch(train_dataset, BATCH_SIZE),
                                 steps_per_epoch=int(len(train_dataset)/BATCH_SIZE)+1,
                                 epochs=EPOCHS,
                                 verbose=2,
                                 max_q_size=1,
                                 validation_data=generate_test_batch(test_dataset, 1),
                                 validation_steps=len(test_dataset),
                                 callbacks=[checkpoint])

    def predict(self, image):
        """
        Outputs a one-shot steering angle prediction, given an input image.

        :param image: the image for which we want to predict the steering angle
        :return: the predicted steering angle
        """
        if type(image) is np.ndarray:
            image_in = np.empty([1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
            image_in[0] = preprocess_input(image)
            steering = self.model.predict(image_in, batch_size=1)[0][0]
            return steering
        else:
            print('Empty image')



