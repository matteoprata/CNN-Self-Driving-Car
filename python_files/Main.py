
# =====================================================================
#  Main.py
# =====================================================================
#
#  Author:         (c) 2019 Antonio Pio Ricciardi & Matteo Prata
#  Created:        January  02, 2019


from Server import Server, NeuralArchitecture
from CNN_Preprocessing import load_data, DATA_PATH
from CNN import CNN


def run(neural_arch, is_trained=False, model_path=None):
    """
    Responsible for starting the program.

    :param neural_arch: defines the neural architecture that we want to use, it's a NeuralArchitecture enumerator.
    :param is_trained: True if it's trained (mandatory to specify the model in 'model_path'), False otherwise (it starts the training). By default it is set to False.
    :param model_path: the path in which the model is located (to specify only if 'is_trained' is True).
    """

    if is_trained and not model_path:
        print("Please specify the model in function call 'run(...)'.")
        return

    elif not is_trained and model_path:
        print("No need to specify the model in function call 'run(...)'. Model ignored.")

    # Start a CNN to train if asked,
    # in any other case handle the Client <> Server communication with the Unity 3D Wold
    if neural_arch == NeuralArchitecture.CNN and not is_trained:
        cnn = CNN()
        train_dataset, test_dataset = load_data(DATA_PATH)
        cnn.train_model(train_dataset, test_dataset)
    else:
        Server(com_type=neural_arch, is_trained_com=is_trained, model_path=model_path).run()


# Train the CNN with the input dataset
# run(NeuralArchitecture.CNN, is_trained=False)

# Execute prediction with the CNN to the Unity 3D World using the specified model
run(NeuralArchitecture.CNN, is_trained=True, model_path="model-ottimo12.h5")
