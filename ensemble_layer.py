import copy
import pdb
from time import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from encoding_helper import LayerEncodingWrapper
from utils import Metadata


# import logging


# logging.getLogger().setLevel(logging.INFO)

class EnsembleLayer:
    """ Class for an ensemble layer
    """

    def __init__(self, clfs, n_cv_folds, n_cls, n_clfs, n_clfs_prev, meta_clf):
        """

        Parameters
        ----------
        clfs : List of algorithms (untrained clfs)

        n_cv_folds : Number of folds used for cross validation

        n_cls : Number of cls

        n_clfs : Number of clfs

        n_clfs_prev: Number of clfs used in previous layer
        (for using metadata)

        meta_clf: Meta-classifier

        """
        self.meta_clf = meta_clf
        self.clfs = clfs
        self.n_cv_folds = n_cv_folds
        self.n_cls = n_cls
        self.n_clfs = n_clfs
        self.n_clfs_prev = n_clfs_prev
        self.trained_models = []
        self.hashtable = {} # To help with fitness evaluation


    def cross_validation(self, X, y, X_org):
        """ Parameters
        ----------
        X : Data from the previous layer, of size (n_inst, n_cls*n_clfs_prev)

        y : Ground truth from the previous layer, of size (n_inst).

        X_org : Original data (n_inst, n_features))

        Returns
        -------
        metadata: Metadata, of size (n_inst, n_cls*n_clfs_out) where
        n_clfs_out is the number of clfs used in the current layer

        """

        clfs = self.clfs
        n_clfs = self.n_clfs
        n_clfs_prev = self.n_clfs_prev
        n_clfs_out = self.n_clfs_out
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs_out = self.n_clfs_out
        clfs_used = self.clfs_used
        features_used = self.features_used
        n_inst = X.shape[0]

        metadata = Metadata(n_inst, n_cls, n_clfs_out)
        kf = KFold(n_splits=self.n_cv_folds)
        kf_split = kf.split(X)
        X_in = np.hstack((X, X_org))
        for train_ids, val_ids in kf_split:
            X_train = X_in[train_ids, :]
            y_train = y[train_ids]
            X_val = X_in[val_ids, :]
            metadata_clfs_idx = 0
            for i_clfs in range(n_clfs):
                if clfs_used[i_clfs] == 0:
                    # No input to this classifier => Not used
                    continue
                new_clfs = copy.deepcopy(self.clfs[i_clfs])
                features_used_i = features_used[i_clfs, :]
                new_clfs.fit(X_train[:, features_used_i == 1], y_train)
                prob = new_clfs.predict_proba(X_val[:, features_used_i == 1])
                if (np.isnan(np.sum(prob)) is True):
                    pdb.set_trace()
                metadata.set(prob, metadata_clfs_idx, val_ids)
                metadata_clfs_idx += 1
            if metadata_clfs_idx != n_clfs_out:
                pdb.set_trace()
        metadata = metadata.get()
        return metadata

    def fit(self, X, y, X_org):
        """ Parameters
        ----------
        X : Data from the previous layer, of size (n_inst, n_cls*n_clfs_prev)

        y : Ground truth from the previous layer, of size (n_inst).

        X_org : Original data (n_inst, n_features))

        Returns
        -------
        None

        """
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_clfs_prev = self.n_clfs_prev
        n_clfs_out = self.n_clfs_out
        clfs_used = self.clfs_used
        features_used = self.features_used

        n_inst = X.shape[0]
        X_in = np.hstack((X, X_org))
        for i_clfs in range(self.n_clfs):
            if clfs_used[i_clfs] == 0:
                continue            
            new_clfs = copy.deepcopy(clfs[i_clfs])            
            features_used_i = features_used[i_clfs, :]
            new_clfs.fit(X_in[:, features_used_i == 1], y)

            self.trained_models.append(new_clfs)

    def predict_metadata(self, X, X_org):
        """ Parameters
        ----------
        X : Data from the previous layer, of size (n_inst, n_cls*n_clfs_prev)

        X_org : Original data (n_inst, n_features))

        Returns
        -------
        None

        """
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_clfs_prev = self.n_clfs_prev
        n_clfs_out = self.n_clfs_out
        clfs_used = self.clfs_used
        features_used = self.features_used
        n_inst = X.shape[0]

        X_in = np.hstack((X, X_org))
        metadata = Metadata(n_inst, n_cls, n_clfs_out)
        metadata_clfs_idx = 0
        for i_clfs in range(self.n_clfs):
            if clfs_used[i_clfs] == 0:
                continue
            features_used_i = features_used[i_clfs, :]
            prob = self.trained_models[metadata_clfs_idx].predict_proba(X_in[:, features_used_i == 1])
            metadata.set(prob, metadata_clfs_idx)
            metadata_clfs_idx += 1
        if metadata_clfs_idx != n_clfs_out:
            pdb.set_trace()
        metadata = metadata.get()
        return metadata

    def fit_and_predict_metadata(self, X_train, y_train, X_val, y_val, X_train_org, X_val_org):
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_clfs_prev = self.n_clfs_prev
        n_clfs_out = self.n_clfs_out
        clfs_used = self.clfs_used
        features_used = self.features_used

        n_inst = X_val.shape[0]
        X_train_in = np.hstack((X_train, X_train_org))
        X_val_in = np.hstack((X_val, X_val_org))
        metadata = Metadata(n_inst, n_cls, n_clfs_out)
        metadata_clfs_idx = 0
        for i_clfs in range(self.n_clfs):
            if clfs_used[i_clfs] == 0:
                continue   
            new_clfs = copy.deepcopy(clfs[i_clfs])
            features_used_i = features_used[i_clfs, :]
            new_clfs.fit(X_train_in[:, features_used_i == 1], y_train)
            prob = new_clfs.predict_proba(X_val_in[:, features_used_i == 1])
            metadata.set(prob, metadata_clfs_idx)
            metadata_clfs_idx += 1
        metadata = metadata.get()
        return metadata

    def run(self, X_train, y_train, X_val, y_val, X_train_org, X_val_org, enc_in, eval_fitness=False):
        """ Run layer

        Parameters
        ----------
        X_train: Train data from previous layer, of size
        (n_train_inst, n_cls*n_clfs_prev)

        y_train: Train labels, of size (n_train_inst)

        X_val: Validation data from previous layer, of size
        (n_val_inst, n_cls*n_clfs_prev)

        y_val: Validation labels, of size (n_val_inst)

        X_train_org: Train data original, of size
        (n_train_inst, n_features)

        X_val_org: Validation data original, of size
        (n_val_inst, n_features)

        enc_in: Binary Numpy array, of size
        ((n_clfs_prev+1)*n_clfs)

        eval_fitness: Boolean variable, denoting whether this function is used by an optimizer
        or not, so that some computations can be saved.

        Returns
        -------
        metadata_train : Train metadata, of size
        (n_train_inst, n_clfs*n_cls)

        metadata_val : Validation metadata, of size
        (n_val_inst,  n_clfs*n_cls)

        pred_val: Prediction on validation, of size (n_val_inst)

        """
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_clfs_prev = self.n_clfs_prev
        n_features = X_train_org.shape[1]
        n_train_inst = X_train.shape[0]
        n_val_inst = X_val.shape[0]

        enc = copy.deepcopy(enc_in)
        if np.sum(enc) == 0:
            # Nothing is chosen => Choose everything
            enc[enc == 0] = 1

        enc_wrapper = LayerEncodingWrapper(enc, n_clfs_prev, n_clfs)
        n_clfs_out = enc_wrapper.num_of_clfs_out()
        clfs_used = enc_wrapper.get_clfs_used()
        features_used = enc_wrapper.get_features_used(n_cls, n_features)

        self.n_clfs_out = n_clfs_out
        self.clfs_used = clfs_used
        self.features_used = features_used
        self.n_features = n_features
        self.trained_models = []

        if (type(self.meta_clf).__name__ == 'SumRuleCombiner') and (eval_fitness==True):
            # Sum rule and we are evaluating fitness for optimizer 
            # so we don't need to perform cross validation
            metadata_train = [] 
            metadata_val = self.fit_and_predict_metadata(X_train, y_train, X_val, y_val, X_train_org, X_val_org)
            model = self.meta_clf.fit(metadata_train, y_train, n_cls, n_clfs_out)
            pred_val = model.predict(metadata_val)
        else:
            metadata_train = self.cross_validation(X_train, y_train, X_train_org)
            self.fit(X_train, y_train, X_train_org)
            metadata_val = self.predict_metadata(X_val, X_val_org)
            model = self.meta_clf.fit(metadata_train, y_train, n_cls, n_clfs_out)
            pred_val = model.predict(metadata_val)
        return (metadata_train, metadata_val, pred_val)

    def fitness(self, X_train, y_train, X_val, y_val, X_train_org, X_val_org, enc):
        """ Calculate fitness given an encoding.
            Used by the optimization routine.

        Parameters
        ----------
        X_train: Train data, of size (n_train_inst, n_features)

        y_train: Train labels, of size (n_train_inst)

        X_val: Validation data, of size (n_val_inst, n_features)

        y_val: Validation labels, of size (n_val_inst)

        X_train_org: Train data original, of size (n_train_inst, n_features)

        X_val_org: Validation data original, of size (n_val_inst, n_features)

        enc: Binary Numpy array, of size (n_clfs)

        Returns
        -------
        acc: Scalar (0 <= acc <= 1)
        """
        
        enc_to_bytes = enc.astype(np.int32).tobytes()
        if enc_to_bytes in self.hashtable:
            return self.hashtable[enc_to_bytes]
        else:        
            (_, _, pred_val) = self.run(X_train, y_train, X_val, y_val,
                                        X_train_org, X_val_org, enc, eval_fitness=True)
            acc = accuracy_score(y_val, pred_val)
            self.hashtable[enc_to_bytes] = acc
            return acc
