# -*- coding: utf-8 -*-
import ipdb
import theano.tensor as T

from vamp import VAMP_smartlearner
from pylearn2.datasets.deep_vamp import VAMP
from utils import Timer
from utils import load_dataset_vamp
from smartlearner import Trainer
from smartlearner.optimizers import SGD
from smartlearner.tasks import stopping_criteria
from smartlearner.tasks import tasks
from smartlearner.tasks import views
from smartlearner.batch_scheduler import MiniBatchScheduler
#from batch_scheduler2 import MiniBatchScheduler
from smartlearner.losses import NegativeLogLikelihood
from smartlearner.update_rules import ConstantLearningRate

from task import NegativeLogLikelihoodLoss


def train_simple_nade():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_dataset_vamp()

        # The target for distribution estimator is the input
        #trainset._targets_shared = trainset.inputs
        #validset._targets_shared = validset.inputs
        #testset._targets_shared = testset.inputs

    with Timer("Creating model"):
        #hidden_size = 50
        model = VAMP_smartlearner()
        # model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        optimizer = SGD(loss=NegativeLogLikelihood(model, trainset))
        optimizer.append_update_rule(ConstantLearningRate(0.0001))

    with Timer("Building trainer"):
        # Train using mini batches of 100 examples
        batch_scheduler = MiniBatchScheduler(trainset, model.batch_size)

        trainer = Trainer(optimizer, batch_scheduler)
        # Train for 10 epochs
        trainer.append_task(stopping_criteria.MaxEpochStopping(1))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        # ipdb.set_trace()
        nll_error_train = NegativeLogLikelihoodLoss(model.get_model_output, trainset)
        nll_error_valid = NegativeLogLikelihoodLoss(model.get_model_output, validset)
        #classif_error_train = views.ClassificationError(model.use, trainset)
        #classif_error_valid = views.ClassificationError(model.use, validset)
        trainer.append_task(tasks.Print("\t Trainset - NLLL: {0:.1%}", nll_error_train.mean))

        results = "\t Trainset - NLL: {0:.1%} ± {1:.1%}"
        results += "\t Validset - NLL: {2:.1%} ± {3:.1%}"
        #results += "\t Trainset - classif_err: {4:.1%} ± {5:.1%}"
        #results += "\t Validset - classif_err: {6:.1%} ± {7:.1%}"

        # , classif_error_train.mean, classif_error_train.stderror, classif_error_valid.mean, classif_error_valid.stderror))
        trainer.append_task(tasks.Print(results, nll_error_train.mean, nll_error_train.stderror, nll_error_valid.mean, nll_error_valid.stderror))

        #trainer.append_task(tasks.Print("\t Trainset - NLL: {0:.1%} ± {1:.1%}", nll_error_train.mean, nll_error_train.stderror))
        #trainer.append_task(tasks.Print("\t Validset - NLL: {0:.1%} ± {1:.1%}", nll_error_valid.mean, nll_error_valid.stderror))
        #trainer.append_task(tasks.Print("\t Trainset - classif_err: {0:.1%} ± {1:.1%}", classif_error_train.mean, classif_error_train.stderror))
        #trainer.append_task(tasks.Print("\t Validset - classif_err: {0:.1%} ± {1:.1%}", classif_error_valid.mean, classif_error_valid.stderror))

    with Timer("Training"):
        trainer.train()

    output_task = tasks.GetOutPut(model.get_last_hiddenlayer_output, testset)
    ttt = output_task.update(None)
    import ipdb
    ipdb.set_trace()

if __name__ == "__main__":
    train_simple_nade()
