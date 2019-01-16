
# =====================================================================
#  CNNCommunicationHandler.py
# =====================================================================
#
#  Author:         (c) 2019 Antonio Pio Ricciardi & Matteo Prata
#  Created:        January  02, 2019

from CNN import CNN
from keras.models import load_model
import cv2


class CNNCommunicationHandler:
    cnn = None
    messages_size = None

    def __init__(self, model_path, messages_size):
        """
        Represents the communication between the trained CNN and the Unity 3D World, for next action prediction.

        :param model_path: the path of the trained model
        :param messages_size: the size in bytes of the message to read
        """

        # load the model and assign it to the CNN
        model = load_model(model_path)
        self.messages_size = messages_size
        self.cnn = CNN(model=model)

    def communication(self, message0, client):
        """
        It handles the communication between the CNN and the Unity 3D World, sending messages and receiving them.

        :param message0: the first message comes from the caller
        :param client: the client object from/to which we receive/send messages
        :return: the next message to analyse in the communication loop
        """

        img_path = message0.decode()
        image = cv2.imread(img_path)

        steering = self.cnn.predict(image)
        client.send(str.encode(str(steering) + '\r\n'))

        message1 = client.recv(self.messages_size)
        return message1
