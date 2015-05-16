from __future__ import print_function

# MLPython datasets wrapper
import os
import sys
import json
import theano
import theano.sandbox.softsign
import numpy as np
import mlpython.datasets.store as mlstore
from time import time

from smartpy import Dataset


def save_dict_to_json_file(path, dictionary):
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': ')))


def load_dict_from_json_file(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())


class Timer():
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


ACTIVATION_FUNCTIONS = {
    "sigmoid": theano.tensor.nnet.sigmoid,
    "hinge": lambda x: theano.tensor.maximum(x, 0.0),
    "softplus": theano.tensor.nnet.softplus,
    "tanh": theano.tensor.tanh,
    "softsign": theano.sandbox.softsign.softsign,
    "brain": lambda x: theano.tensor.maximum(theano.tensor.log(theano.tensor.maximum(x + 1, 1)), 0.0)
}


class WeightsInitializer(object):
    def __init__(self, random_seed=None):
        self.rng = np.random.mtrand.RandomState(random_seed)

    def _init_range(self, dim):
        return np.sqrt(6. / (dim[0] + dim[1]))

    def uniform(self, dim):
        init_range = self._init_range(dim)
        return np.asarray(self.rng.uniform(low=-init_range, high=init_range, size=dim), dtype=theano.config.floatX)

    def zeros(self, dim):
        return np.zeros(dim, dtype=theano.config.floatX)

    def diagonal(self, dim):
        W_values = self.zeros(dim)
        np.fill_diagonal(W_values, 1)
        return W_values

    def orthogonal(self, dim):
        max_dim = max(dim)
        return np.linalg.svd(self.uniform((max_dim, max_dim)))[2][:dim[0], :dim[1]]

    def gaussian(self, dim):
        return np.asarray(self.rng.normal(loc=0, scale=self._init_range(dim), size=dim), dtype=theano.config.floatX)


# List of supported datasets
DATASETS = ['adult',
            'binarized_mnist',
            'connect4',
            'dna',
            'mushrooms',
            'mnist'
            'nips',
            'ocr_letters',
            'rcv1',
            'rcv2_russ',
            'web']


def load_dataset(dataset_name):
    #Temporary patch until we build the dataset manager
    dataset_npy = os.path.join(os.environ['MLPYTHON_DATASET_REPO'], dataset_name, 'data.npz')
    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(os.path.join(os.environ['MLPYTHON_DATASET_REPO'], dataset_name)):
            mlstore.download(dataset_name)

        if dataset_name in mlstore.classification_names:
            trainset, validset, testset = mlstore.get_classification_problem(dataset_name)
            trainset, validset, testset = zip(*trainset), zip(*validset), zip(*testset)
            trainset_inputs, trainset_targets = np.array(trainset[0]), np.array(trainset[1], ndmin=2).T
            validset_inputs, validset_targets = np.array(validset[0]), np.array(validset[1], ndmin=2).T
            testset_inputs, testset_targets = np.array(testset[0]), np.array(testset[1], ndmin=2).T
        elif dataset_name in mlstore.distribution_names:
            trainset, validset, testset = mlstore.get_distribution_problem(dataset_name)
            trainset_inputs, trainset_targets = np.array(trainset), np.zeros((len(trainset), 0))
            validset_inputs, validset_targets = np.array(validset), np.zeros((len(validset), 0))
            testset_inputs, testset_targets = np.array(testset), np.zeros((len(testset), 0))
        else:
            print("Not supported type of dataset!")
            return

        np.savez(dataset_npy,
                 trainset_inputs=trainset_inputs, trainset_targets=trainset_targets,
                 validset_inputs=validset_inputs, validset_targets=validset_targets,
                 testset_inputs=testset_inputs, testset_targets=testset_targets)

    data = np.load(dataset_npy)
    trainset = Dataset(data['trainset_inputs'].astype(theano.config.floatX), data['trainset_targets'].astype(theano.config.floatX))
    validset = Dataset(data['validset_inputs'].astype(theano.config.floatX), data['validset_targets'].astype(theano.config.floatX))
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), data['testset_targets'].astype(theano.config.floatX))

    return trainset, validset, testset
