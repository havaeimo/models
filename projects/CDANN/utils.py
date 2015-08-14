from __future__ import print_function

# MLPython datasets wrapper
import os
import sys
import json
import theano
import theano.sandbox.softsign
import numpy as np
from time import time
from pylearn2.datasets.deep_vamp import VAMP, FullImageDataset
from smartlearner import Dataset
from dataset2 import DatasetWithTargetDomain
import theano.tensor as T
import ipdb
DATASETS_ENV = 'DATASETS'


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
        print("{:.2f} sec.".format(time() - self.start))


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


def load_dataset_cdann():
    # Temporary patch until we build the dataset manager
    image_size = [128, 64]
    train_s = VAMP(start=0, stop=10000, image_resize=image_size, toronto_prepro=True, read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_VIRTUAL')
    train_t = VAMP(start=0, stop=10000, image_resize=image_size, toronto_prepro=True, read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_REAL')
    valid = VAMP(start=0, stop=300, image_resize=image_size, toronto_prepro=True, read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL')
    test = VAMP(start=2000, stop=3000, image_resize=image_size, toronto_prepro=True, read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL')

    #datasets_repo = os.environ.get(DATASETS_ENV, './datasets')
    # if not os.path.isdir(datasets_repo):
    #    os.mkdir(datasets_repo)
    image_s = train_s.X
    target_s_f = np.argmax(train_s.y, axis=1).reshape(-1, 1)  # target is required to be a matrix
    target_s_d = np.ones(target_s_f.shape)

    image_t = train_t.X
    target_t_d = np.zeros(target_s_f.shape)

    image_valid = valid.X
    target_valid_f = np.argmax(valid.y, axis=1).reshape(-1, 1)  # target is required to be a matrix

    image_test = test.X
    target_test_f = np.argmax(test.y, axis=1).reshape(-1, 1)  # target is required to be a matrix

    trainset1 = DatasetWithTargetDomain(inputs=image_s.astype(theano.config.floatX), targets1=target_s_f.astype(
        theano.config.floatX), targets2=target_s_d.astype(theano.config.floatX), name="trainset1")
    trainset2 = Dataset(inputs=image_t.astype(theano.config.floatX), targets=target_t_d.astype(theano.config.floatX), name="trainset2")

    trainset_f = Dataset(image_s.astype(theano.config.floatX), targets=target_s_f.astype(theano.config.floatX), name="trainset_f")

    validset = Dataset(image_valid.astype(theano.config.floatX), targets=target_valid_f.astype(theano.config.floatX), name="validset")
    testset = Dataset(image_test.astype(theano.config.floatX), targets=target_test_f.astype(theano.config.floatX), name="testset")

    trainset2.symb_inputs = T.tensor4(name=trainset2.name)  # TODO: Removed this once issue #32 in smartlearner is fixed.
    trainset_f.symb_inputs = T.tensor4(name=trainset_f.name)  # TODO: Removed this once issue #32 in smartlearner is fixed.
    validset.symb_inputs = T.tensor4(name=validset.name)  # TODO: Removed this once issue #32 in smartlearner is fixed.
    testset.symb_inputs = T.tensor4(name=testset.name)  # TODO: Removed this once issue #32 in smartlearner is fixed.

    return trainset1, trainset2, trainset_f, validset, testset


def load_dataset_vamp():
    # Temporary patch until we build the dataset manager
    image_size = [128, 64]
    nb_channels = 3
    #train_s = VAMP(start=0,stop=100,image_resize=image_size,toronto_prepro=True,read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_VIRTUAL')
    #valid =   VAMP(start=0,stop=100,image_resize=image_size,toronto_prepro=True,read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/PN_REAL')
    #test = VAMP(start=20,stop=3000,image_resize=image_size,toronto_prepro=True,read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL')
    train_s = VAMP(start=100, stop=150, image_resize=image_size, toronto_prepro=True, read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL')
    valid = VAMP(start=200, stop=250, image_resize=image_size, toronto_prepro=True, read='/home/local/USHERBROOKE/havm2701/data/Data/Deep_VAMP/TRAIN_REAL_PN_VIRTUAL')
    test = FullImageDataset('/home/local/USHERBROOKE/havm2701/data/Data/DBFrames')
    image_s = train_s.X.reshape((train_s.X.shape[0], image_size[0], image_size[1], nb_channels))
    #image_s = np.transpose(image_s,(0,3,1,2))
    target_s_f = np.argmax(train_s.y, axis=1).reshape(-1, 1)  # target is required to be a matrix

    image_valid = valid.X.reshape((valid.X.shape[0], image_size[0], image_size[1], nb_channels))
    #image_valid = np.transpose(image_valid,(0,3,1,2))
    target_valid_f = np.argmax(valid.y, axis=1).reshape(-1, 1)  # target is required to be a matrix

    image_test = test.X[:50, ...]
    #image_test = np.transpose(image_test,(0,3,1,2))
    target_test_f = test.y[:50].reshape(-1, 1)  # target is required to be a matrix
    trainset = Dataset(inputs=image_s.astype(theano.config.floatX), targets=target_s_f.astype(theano.config.floatX), name="trainset")
    validset = Dataset(inputs=image_valid.astype(theano.config.floatX), targets=target_valid_f.astype(theano.config.floatX), name="validset")
    testset = Dataset(inputs=image_test.astype(theano.config.floatX), targets=target_test_f.astype(theano.config.floatX), name="testset")
    return trainset, validset, testset


def negative_log_likelihood(model_output, y):

    #p_y_given_x_shuff = p_y_given_x.dimshuffle(0,2,3,1)
    #p_y_given_x_flat = p_y_given_x_shuff.flatten(2)
    #y = T.cast(y[:, 0], dtype="int32")
    # return -T.mean(T.log(p_y_given_x_flat)[T.arange(y.shape[0]), y])
    nll = -T.log(model_output)
    indices = T.cast(y[:, 0], dtype="int32")  # Targets are floats.
    selected_nll = nll[T.arange(y.shape[0]), indices]
    return T.mean(selected_nll)


def negative_log_likelihood_array(model_output, y):

    nll = -T.log(model_output)
    indices = T.cast(y[:, 0], dtype="int32")  # Targets are floats.
    selected_nll = nll[T.arange(y.shape[0]), indices]
    return selected_nll
