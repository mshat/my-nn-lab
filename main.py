from models import Layer, NN
from files_io import load_trained_nn

from numbers_recognition import NumRecTrainer, NumRecLaboratory


DATASET_FILENAME = "mnist.npz"


def experiment1():
    trainer = NumRecTrainer(epochs=3)
    laboratory = NumRecLaboratory(trainer)

    nn = NN([Layer(784), Layer(20), Layer(10),])
    # nn = NN([Layer(784), Layer(50), Layer(10), ])
    # nn = NN([Layer(784), Layer(10)])

    nn_name = f"experiment1_trained_nn_{str(nn)}"

    dataset = laboratory.load_first_n_samples(55000)
    test_dataset = laboratory.load_last_n_samples(5000)

    trained_nn = laboratory.train(nn, dataset, nn_name)

    trained_nn = load_trained_nn(nn_name)
    laboratory.test_nn(test_dataset, trained_nn)
    laboratory.custom_test(trained_nn)
    laboratory.show_synapse(trained_nn, layer_index=0, sinaps_index=0)
    laboratory.show_layer_synapses(trained_nn, layer_index=0)


def main():
    experiment1()


if __name__ == "__main__":
    main()

