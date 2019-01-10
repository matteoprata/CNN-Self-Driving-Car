
# =====================================================================
#  Server.py
# =====================================================================
#
#  Author:         (c) 2019 Antonio Pio Ricciardi & Matteo Prata
#  Created:        January  02, 2019


import socket
from RLNNCommunicationHandler import RLNNCommunicationHandler
from CNNCommunicationHandler import CNNCommunicationHandler
from enum import Enum

MESSAGE_BYTE_SIZE = 1024


class NeuralArchitecture(Enum):
    """
    Represents the two possible neural architectures allowed by our 3D models in Unity.
    """
    CNN = 1
    RLNN = 2


class Server:

    host = 'localhost'
    port = 50000
    backlog = 5
    socket = None
    is_trained_com = None
    com_type = None
    model_path = None
    ch = None

    def __init__(self, com_type, is_trained_com, model_path):
        """
        Represents the Server which sends and receives the information to/from the Unity 3D World.

        :param com_type: the kind of client server communication to allow, to specify with the enumerator 'NeuralArchitecture'
        :param is_trained_com: True if it's trained (specify the model in 'model_path'), False otherwise (only in RLNN).
        :param model_path: the path in which the model is located (to specify only if 'is_trained' is True).
        """

        self.is_trained_com = is_trained_com
        self.model_path = model_path
        self.com_type = com_type

        # Setup the socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(self.backlog)

        # How to handle the communication, depending on the kind of neural network in use
        if self.com_type == NeuralArchitecture.RLNN:
            # self.ch = RLNNCommunicationHandler(self.is_trained_com, MESSAGE_BYTE_SIZE) # public soon
            pass

        elif self.com_type == NeuralArchitecture.CNN:
            self.ch = CNNCommunicationHandler(self.model_path, MESSAGE_BYTE_SIZE)

    def run(self):
        """
        It runs the Server.
        """

        print('Waiting for client to connect...')
        client, _ = self.socket.accept()
        print('Client connected')

        # Waits for the first message and starts the comunication
        message0 = client.recv(MESSAGE_BYTE_SIZE)
        while 1:
            message0 = self.ch.communication(message0, client)
