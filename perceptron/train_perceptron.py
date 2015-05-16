# -*- coding: utf-8 -*-

import theano.tensor as T

from perceptron import Perceptron
from utils import load_dataset
from utils import Timer
from smartpy import SGD
from smartpy import Trainer, tasks
from smartpy.update_rules import LearningRate


def train_simple_perceptron():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_dataset('mnist')

    with Timer("Creating model"):
        input_size = trainset.input_size
        # TODO: We should the number of different targets in the dataset,
        #       but I'm not sure how to do it right (keep in mind the regression?).
        output_size = 10
        model = Perceptron(input_size, output_size)
        model.initialize()  # By default, uniform initialization.

    with Timer("Making loss symbolic graph"):
        def mean_nll(input, target):
            probs = model.fprop(input)
            nll = -T.log(probs)
            indices = T.cast(target[:, 0], dtype="int32")  # Targets are floats.
            selected_nll = nll[T.arange(target.shape[0]), indices]
            return T.mean(selected_nll)

    with Timer("Building optimizer"):
        optimizer = SGD(model, loss_fct=mean_nll, dataset=trainset, batch_size=100)
        optimizer.append_update_rule(LearningRate(0.0001))

    with Timer("Building trainer"):
        trainer = Trainer(optimizer)

        # Train for 10 epochs
        trainer.append_stopping_criterion(tasks.MaxEpochStopping(10))

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())

        # Print mean/stderror of classification errors.
        classif_error = tasks.ClassificationError(model.use, validset)
        trainer.append_task(tasks.Print("Validset - Classif error: {0:.1%} ± {1:.1%}", classif_error.mean, classif_error.stderror))

    with Timer("Training"):
        trainer.train()


if __name__ == "__main__":
    train_simple_perceptron()
