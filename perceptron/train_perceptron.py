# -*- coding: utf-8 -*-

import theano.tensor as T

from models.perceptron import Perceptron
from models.utils import load_mnist
from models.utils import Timer

from smartpy import Trainer, tasks
from smartpy.optimizers import SGD
from smartpy.update_rules import ConstantLearningRate
from smartpy.interfaces.loss import NegativeLogLikelihood as NLL
from smartpy.batch_scheduler import MiniBatchScheduler


def train_simple_perceptron():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_mnist()

    with Timer("Creating model"):
        # TODO: We should the number of different targets in the dataset,
        #       but I'm not sure how to do it right (keep in mind the regression?).
        output_size = 10
        model = Perceptron(trainset.input_size, output_size)
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        optimizer = SGD(loss=NLL(model, trainset))
        optimizer.append_update_rule(ConstantLearningRate(0.0001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        batch_scheduler = MiniBatchScheduler(trainset, 100)
        stopping_criterion = tasks.MaxEpochStopping(10)

        trainer = Trainer(optimizer, batch_scheduler, stopping_criterion=stopping_criterion)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        classif_error = tasks.ClassificationError(model.use, validset)
        trainer.append_task(tasks.Print("Validset - Classif error: {0:.1%} ± {1:.1%}", classif_error.mean, classif_error.stderror))

    with Timer("Training"):
        trainer.train()


if __name__ == "__main__":
    train_simple_perceptron()
