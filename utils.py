import pdb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Metadata:
    """ Wrapper class for metadata
    """

    def __init__(self, n_inst, n_cls, n_clfs):
        """

        Parameters
        ----------
        n_inst: Number of instances

        n_cls : Number of classes

        n_clfs : Number of classifiers
        """
        self.n_inst = n_inst
        self.n_cls = n_cls
        self.n_clfs = n_clfs
        self.metadata = np.zeros((n_inst, n_cls * n_clfs))

    def get(self):
        """ Get metadata

        Parameters
        ----------
        None

        Returns
        -------
        metadata: Numpy array, of size (n_inst, n_cls*n_clfs)
        """
        return self.metadata

    def get_part(self, clfs_idx):
        """ Get part of metadata relating to classifier clfs_idx

        Parameters
        -----------
        clfs_idx: Classifier index

        Returns
        -------
        prob: Numpy array, of size (n_inst, n_cls)
        """
        n_inst = self.n_inst
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        return copy.deepcopy(self.metadata[:, clfs_idx*n_cls: (clfs_idx+1)*n_cls])

    def set(self, prob, clfs_idx, inst_ids=None):
        """ Set metadata

        Parameters
        ----------
        prob: New 2D Numpy array data to be added

        clfs_idx: Index of current classifier

        inst_ids: If None, add data to all indices, else add only to inst_ids

        Returns
        -------
        None
        """
        n_inst = self.n_inst
        n_cls = self.n_cls
        n_clfs = self.n_clfs

        start_idx = clfs_idx * n_cls
        end_idx = (clfs_idx + 1) * n_cls
        if inst_ids is None:
            self.metadata[:, start_idx:end_idx] = prob
        else:
            self.metadata[inst_ids, start_idx:end_idx] = prob

def collect_result(y_test, y_pred):
    result = {}
    result['acc'] = accuracy_score(y_test, y_pred)
    result['precision'] = precision_score(y_test, y_pred, average="macro")
    result['recall'] = recall_score(y_test, y_pred, average="macro")
    result['f1'] = f1_score(y_test, y_pred, average="macro")
    return result

def main():
    n_inst = 3
    n_clfs = 2
    n_cls = 4
    metadata = Metadata(n_inst, n_cls, n_clfs)
    inst_ids = np.array([0, 1], dtype=np.int32)
    prob = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.1, 0.0, 0.0],
    ])
    clfs_idx = 0
    metadata.set(prob, clfs_idx, inst_ids=inst_ids)
    temp = metadata.get()
#    prob = np.array([
#        [0.1, 0.2, 0.3, 0.4, 0.6, 0.1, 0.1, 0.2],
#        [0.9, 0.1, 0.0, 0.0, 0.7, 0.1, 0.1, 0.1],
#    ])
#        [0.3, 0.3, 0.3, 0.1, 0.1, 0.5, 0.2, 0.2]

    pdb.set_trace()

if __name__ == '__main__':
    main()
