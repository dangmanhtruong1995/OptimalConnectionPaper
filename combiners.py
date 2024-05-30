import numpy as np
import scipy


class GenericCombiner(object):
    """
    Inherit from this class to create your own combiner
    """

    def __init__(self):
        pass

    def fit(self, metadata, labels, n_cls, n_clfs):
        raise NotImplementedError

    def predict(self, metadata):
        raise NotImplementedError


class SumRuleCombiner(GenericCombiner):
    def __init__(self):
        pass

    def fit(self, metadata, labels, n_cls, n_clfs):
        self.n_cls = n_cls
        self.n_clfs = n_clfs
        return self

    def predict(self, metadata):
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_inst = metadata.shape[0]
        prob = np.zeros((n_inst, n_cls))
        idx = 0
        for i in range(n_clfs):
            prob += metadata[:, idx: idx + n_cls]
            idx += n_cls
        pred = np.argmax(prob, axis=1) + 1  # Labels start from 1
        return pred

