import theano
from smartlearner import Dataset


class DatasetWithTargetDomain(Dataset):

    def __init__(self, inputs, targets1, targets2, name="dataset"):
        #super(DatasetWithTargetDomain, self).__init__(inputs, targets1, name)
        self.name = name
        self.inputs = inputs
        self.targets1 = targets1
        self.symb_inputs = theano.tensor.matrix(name=self.name)
        self.symb_targets1 = None if targets1 is None else theano.tensor.matrix(name=self.name + '_target1')
        self.targets2 = targets2
        self.symb_targets2 = None if targets2 is None else theano.tensor.matrix(name=self.name + '_target2')

    @property
    def targets1(self):
        return self.targets

    @targets1.setter
    def targets1(self, value):
        self.targets = value

    @property
    def targets2(self):
        return self._targets2_shared

    @targets2.setter
    def targets2(self, value):
        if value is not None:
            self._targets2_shared = theano.shared(value, name=self.name + "_targets2", borrow=True)
        else:
            self._targets2_shared = None

    @property
    def target2_size(self):
        if self.targets2 is None:
            return 0
        else:
            return len(self.targets2.get_value()[0])
