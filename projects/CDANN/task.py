# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from utils import negative_log_likelihood_array
import theano
import theano.tensor as T

from smartlearner.interfaces.task import Task
from smartlearner.tasks.views import View, ItemGetter


class UpdateEpochNumberOfMyLossTask(Task):

    def __init__(self, loss):
        super(UpdateEpochNumberOfMyLossTask, self).__init__()
        self.loss = loss

    def post_epoch(self, status):
        self.loss.epoch_count.set_value(status.current_epoch)


class ClassificationError(View):

    def __init__(self, predict_fct, dataset):
        super(ClassificationError, self).__init__()

        batch_size = 50  # Internal buffer
        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = T.tensor4('input')
        target = T.matrix('target')
        out = predict_fct(input)

        import theano.printing as printing
        target = printing.Print('target')(target)
        classification_errors = T.neq(out, target)

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}
        self.compute_classification_error = theano.function([no_batch],
                                                            classification_errors,
                                                            givens=givens,
                                                            name="compute_classification_error")


class NegativeLogLikelihoodLoss(View):

    def __init__(self, predict_fct, dataset):
        super(NegativeLogLikelihoodLoss, self).__init__()
        batch_size = 100  # Internal buffer
        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = T.tensor4('input')
        target = T.matrix('target')
        classification_errors = negative_log_likelihood_array(predict_fct(input), target)

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size],
                  target: dataset.targets[no_batch * batch_size:(no_batch + 1) * batch_size]}
        self.compute_classification_error = theano.function([no_batch],
                                                            classification_errors,
                                                            givens=givens,
                                                            name="compute_NLL")

    def update(self, status):
        classif_errors = []
        for i in range(self.nb_batches):
            classif_errors.append(self.compute_classification_error(i))

        classif_errors = np.concatenate(classif_errors)
        return classif_errors.mean(), classif_errors.std(ddof=1) / np.sqrt(len(classif_errors))

    @property
    def mean(self):
        return ItemGetter(self, attribute=0)

    @property
    def stderror(self):
        return ItemGetter(self, attribute=1)


class GetOutPut(View):

    def __init__(self, predict_fct, dataset):
        super(GetOutPut, self).__init__()
        batch_size = 100  # Internal buffer
        self.nb_batches = int(np.ceil(len(dataset) / batch_size))

        input = T.tensor4('input')

        get_output = predict_fct(input)

        no_batch = T.iscalar('no_batch')
        givens = {input: dataset.inputs[no_batch * batch_size:(no_batch + 1) * batch_size]
                  }
        self.get_output = theano.function([no_batch],
                                          get_output,
                                          givens=givens,
                                          name="getoutput")

    def update(self, status):
        output = []
        for i in range(self.nb_batches):
            output.append(self.get_output(i))

        output = np.concatenate(output)
        return output
