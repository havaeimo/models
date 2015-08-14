from collections import OrderedDict
from utils import negative_log_likelihood
from theano import tensor as T

from smartlearner.interfaces.loss import Loss


class LossWithTargetDomain(Loss):

    def __init__(self, model, dataset1, dataset2):

        #import theano
        #import numpy as np
        #theano.config.compute_test_value = 'warn'
        self.model = model
        self.dataset2 = dataset2
        self.dataset1 = dataset1
        self.target1 = dataset1.symb_targets1
        self.target2 = dataset1.symb_targets2
        self.targett = dataset2.symb_targets

        #self.dataset1.symb_inputs.tag.test_value = self.dataset1.inputs.get_value()[:5]
        #self.dataset2.symb_inputs.tag.test_value = self.dataset2.inputs.get_value()[:5]
        #self.dataset1.symb_targets1.tag.test_value = self.dataset1.targets1.get_value()[:5]
        #self.dataset1.symb_targets2.tag.test_value = self.dataset1.targets2.get_value()[:5]
        #self.dataset2.symb_targets.tag.test_value = self.dataset2.targets.get_value()[:5]

        # dbg()

    def get_graph_output(self):
        return self._loss_function(self.model.get_model_output_classif(self.dataset1.symb_inputs), self.model.get_model_output_domain(self.dataset1.symb_inputs), self.model.get_model_output_domain(self.dataset2.symb_inputs))

    def _loss_function(self, out_s_f, out_s_d, out_t_d):
        #import theano.printing as printing
        #self.target1 = printing.Print('self.target1')(self.target1)
        Lf_s = negative_log_likelihood(out_s_f, self.target1)
        Ld_t = negative_log_likelihood(out_t_d, self.targett)
        Ld_s = negative_log_likelihood(out_s_d, self.target2)
        gamma = 10
        p = 10
        lambda_p = 2 / (1 + T.exp(-gamma * p)) - 1
        cost = Lf_s - lambda_p * (Ld_s + Ld_t)
        return cost

    def get_gradients(self):

        gparams = T.grad(self.get_graph_output(), list(self.model.parameters.values()))
        gparams[-2:] = -1 * gparams[-2:]
        gradients = dict(zip(self.model.parameters.values(), gparams))
        return gradients, OrderedDict()
