from smartlearner.batch_scheduler import MiniBatchScheduler


class MiniBatchSchedulerWithTargetDomain(MiniBatchScheduler):

    def __init__(self, dataset1, dataset2, batch_size):
        super(MiniBatchSchedulerWithTargetDomain, self).__init__(dataset1, batch_size)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    @property
    def givens(self):
        start = self.shared_batch_count * self._shared_batch_size
        end = (self.shared_batch_count + 1) * self._shared_batch_size

        return {self.dataset1.symb_inputs: self.dataset1.inputs[start:end],
                self.dataset1.symb_targets1: self.dataset1.targets1[start:end],
                self.dataset1.symb_targets2: self.dataset1.targets2[start:end],
                self.dataset2.symb_inputs: self.dataset2.inputs[start:end],
                self.dataset2.symb_targets: self.dataset2.targets[start:end]
                }
