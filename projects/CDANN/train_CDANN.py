# -*- coding: utf-8 -*-
import ipdb
import theano.tensor as T

from cdann import CDANN
from pylearn2.datasets.deep_vamp import VAMP
from utils import Timer
from utils import load_dataset_cdann
from smartlearner import Trainer
from smartlearner.tasks import stopping_criteria
from smartlearner.tasks import tasks
from smartlearner.tasks import views
from smartlearner.optimizers import SGD
#from smartlearner.batch_scheduler import MiniBatchScheduler
from batch_scheduler2 import MiniBatchSchedulerWithTargetDomain
from loss import LossWithTargetDomain
from smartlearner.update_rules import DecreasingLearningRate, ConstantLearningRate

from task import NegativeLogLikelihoodLoss


def train_simple_nade():
    with Timer("Loading dataset"):
        trainset1, trainset2, trainset_f, validset, testset = load_dataset_cdann()
        # The target for distribution estimator is the input
        #trainset._targets_shared = trainset.inputs
        #validset._targets_shared = validset.inputs
        #testset._targets_shared = testset.inputs

    with Timer("Creating model"):
        #hidden_size = 50
        model = CDANN()
        # model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        loss = LossWithTargetDomain(model, trainset1, trainset2)
        optimizer = SGD(loss=loss)
        optimizer.append_update_rule(ConstantLearningRate(0.00001))

    with Timer("Building trainer"):
        # Train using mini batches of 100 examples
        batch_scheduler = MiniBatchSchedulerWithTargetDomain(trainset1, trainset2, 100)

        trainer = Trainer(optimizer, batch_scheduler)
        # Train for 10 epochs
        trainer.append_task(stopping_criteria.MaxEpochStopping(10))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())
        # save epoch count
        #new_task = tasks.UpdateEpochNumberOfMyLossTask(loss)
        # trainer.append_task(new_task)

        # Print mean/stderror of classification errors.
        # ipdb.set_trace()
        nll_error_train = NegativeLogLikelihoodLoss(model.get_model_output_classif, trainset_f)

        #trainer.append_task(tasks.Print("\t Trainset - NLLL: {0:.1%}", nll_error_train.mean))
        trainer.append_task(tasks.Print("\tTrainset - NLL: {0:.1%} ± {1:.1%}", nll_error_train.mean, nll_error_train.stderror))
        classif_error_train = views.ClassificationError(model.use, trainset_f)
        trainer.append_task(tasks.Print("\t Trainset - classif_err: {0:.1%} ± {1:.1%}", classif_error_train.mean, classif_error_train.stderror))
        classif_error_valid = views.ClassificationError(model.use, validset)
        trainer.append_task(tasks.Print("\t Validset - classif_err: {0:.1%} ± {1:.1%}", classif_error_valid.mean, classif_error_valid.stderror))

    with Timer("Training"):
        trainer.train()


if __name__ == "__main__":
    train_simple_nade()
