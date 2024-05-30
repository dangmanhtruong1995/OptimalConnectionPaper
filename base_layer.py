import copy

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from encoding_helper import BaseEncodingWrapper
from utils import Metadata


class BaseLayer:
    """ Class for the base ensemble layer
    """

    def __init__(self, clfs, n_cv_folds, n_cls, n_clfs, meta_clf,
            precompute_metadata_train=None, precompute_metadata_val=None):
        """

        Parameters
        ----------
        clfs : List of algorithms (untrained clfs)

        n_cv_folds : Number of folds used for cross validation

        n_cls : Number of cls

        n_clfs : Number of clfs

        meta_clf: Meta-classifier

        precompute_metadata_train: Precompute metadata of train set(for faster optimization routine)

        precompute_metadata_val: Precompute metadata of validation set
        """

        self.clfs = clfs
        self.n_cv_folds = n_cv_folds
        self.n_cls = n_cls
        self.n_clfs = n_clfs
        self.trained_models = []
        self.meta_clf = meta_clf
        self.precompute_metadata_train = precompute_metadata_train
        self.precompute_metadata_val = precompute_metadata_val

    def cross_validation(self, X, y):
        """ Apply K-folds cross validation on (X, y)
            to obtain the metadata

        Parameters
        ----------
        X : features, of size (n_inst, n_features)

        y : labels, of size (n_inst)

        Returns
        -------
        metadata : Numpy array of size (n_inst, n_cls*n_clfs)
        """

        enc_wrapper = self.enc_wrapper
        n_clfs_out = self.n_clfs_out
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_clfs_out = self.n_clfs_out
        precompute_metadata = self.precompute_metadata_train
        n_inst = X.shape[0]
        metadata = Metadata(n_inst, n_cls, n_clfs_out)

        kf = KFold(n_splits=n_cv_folds)
        kf_split = kf.split(X)

        if precompute_metadata is None:
            for train_ids, test_ids in kf_split:
                X_train = X[train_ids, :]
                y_train = y[train_ids]
                X_test = X[test_ids, :]
                metadata_clfs_idx = 0

                for i_clfs in range(n_clfs):
                    if enc_wrapper.clfs_used(i_clfs) == 0:
                        continue
                    new_clfs = copy.deepcopy(clfs[i_clfs])
                    new_clfs.fit(X_train, y_train)
                    prob = new_clfs.predict_proba(X_test)
                    metadata.set(prob, metadata_clfs_idx, test_ids)
                    metadata_clfs_idx += 1
        else:
            all_clfs_idx = 0
            metadata_clfs_idx = 0
            for i_clfs in range(n_clfs):
                if enc_wrapper.clfs_used(i_clfs) == 0:
                    all_clfs_idx += 1
                    continue
                prob = precompute_metadata[:, all_clfs_idx*n_cls: (all_clfs_idx+1)*n_cls]
                metadata.set(prob, metadata_clfs_idx)
                metadata_clfs_idx += 1
                all_clfs_idx += 1

        return metadata

    def fit(self, X, y):
        """ Train all clfs using X, y
            --> trained_models used to predict test set
        Parameters
        ----------
        X : features, of size (n_inst, n_features)

        y : label, of size (n_inst) (index starts from 1)

        Returns
        -------
        None
        """
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        enc_wrapper = self.enc_wrapper
        n_clfs_out = self.n_clfs_out
        n_inst = X.shape[0]
        precompute_metadata_val = self.precompute_metadata_val

        if precompute_metadata_val is None:
            for i_clfs in range(n_clfs):
                if enc_wrapper.clfs_used(i_clfs) == 0:
                    continue
                new_clfs = copy.deepcopy(clfs[i_clfs])
                new_clfs.fit(X, y)
                self.trained_models.append(new_clfs)

    def predict_metadata(self, X):
        """ Use trained_models to predict probabilities for X

        Parameters
        ----------
        X : unlabeled data, of size (n_inst, n_features)

        Returns
        -------
        metadata : Numpy array of size (n_inst, n_clfs*n_cls)
        """
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        enc_wrapper = self.enc_wrapper
        n_clfs_out = self.n_clfs_out
        n_inst = X.shape[0]
        precompute_metadata = self.precompute_metadata_val

        metadata = Metadata(n_inst, n_cls, n_clfs_out)
        if precompute_metadata is None:
            metadata_clfs_idx = 0
            for i_clfs in range(self.n_clfs):
                if enc_wrapper.clfs_used(i_clfs) == 0:
                    continue
                prob = self.trained_models[metadata_clfs_idx].predict_proba(X)
                metadata.set(prob, metadata_clfs_idx)
                metadata_clfs_idx += 1
        else:
            all_clfs_idx = 0
            metadata_clfs_idx = 0
            for i_clfs in range(n_clfs):
                if enc_wrapper.clfs_used(i_clfs) == 0:
                    all_clfs_idx += 1
                    continue
                prob = precompute_metadata[:, all_clfs_idx*n_cls: (all_clfs_idx+1)*n_cls]
                metadata.set(prob, metadata_clfs_idx)
                metadata_clfs_idx += 1
                all_clfs_idx += 1

        metadata = metadata.get()
        return metadata

    def run(self, X_train, y_train, X_val, y_val, enc_in, eval_fitness=False):
        """ Run layer

        Parameters
        ----------
        X_train: Train data, of size (n_train_inst, n_features)

        y_train: Train labels, of size (n_train_inst)

        X_val: Validation data, of size (n_val_inst, n_features)

        y_val: Validation labels, of size (n_val_inst)

        enc_in: Binary Numpy array (n_clfs)

        eval_fitness: Boolean variable, denoting whether this function is used by an optimizer
        or not, so that some computations can be saved.

        Returns
        -------
        metadata_train : Numpy array of size (n_train_inst, n_clfs*n_cls)

        metadata_val : Numpy array of size (n_val_inst, n_clfs*n_cls)

        pred_val: Numpy array of size (n_val_inst)

        """

        n_cls = self.n_cls
        n_clfs = self.n_clfs
        meta_clf = self.meta_clf
        enc = copy.deepcopy(enc_in)
        if int(np.sum(enc)) == 0:
            # No chosen classifier => All is chosen
            enc[enc == 0] = 1
        enc_wrapper = BaseEncodingWrapper(enc, n_clfs)
        n_clfs_out = enc_wrapper.num_of_clfs_used()
        self.n_clfs_out = n_clfs_out
        self.enc_wrapper = enc_wrapper

        if (type(self.meta_clf).__name__ == 'SumRuleCombiner') and (eval_fitness is True):
            # Sum rule and we are evaluating fitness for optimizer
            # so we don't need to perform cross validation
            metadata_train = []
        else:
            metadata_train = self.cross_validation(X_train, y_train).get()
        self.fit(X_train, y_train)
        metadata_val = self.predict_metadata(X_val)
        model = meta_clf.fit(metadata_train, y_train, n_cls, n_clfs_out)
        pred_val = model.predict(metadata_val)
        return (metadata_train, metadata_val, pred_val)

    def fitness(self, X_train, y_train, X_val, y_val, enc):
        """ Calculate fitness given an encoding.
            Used by the optimization routine.

        Parameters
        ----------
        X_train: Train data, of size (n_train_inst, n_features)

        y_train: Train labels, of size (n_train_inst)

        X_val: Validation data, of size (n_val_inst, n_features)

        y_val: Validation labels, of size (n_val_inst)

        enc: Binary Numpy array, of size (n_clfs)

        Returns
        -------
        acc: Scalar (0 <= acc <= 1)
        """
        (_, _, pred_val) = self.run(X_train, y_train, X_val, y_val, enc, eval_fitness=True)
        acc = accuracy_score(y_val, pred_val)
        return acc
