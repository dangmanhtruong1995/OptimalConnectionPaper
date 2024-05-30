import pdb
import numpy as np
import copy
import numbers

class BaseEncodingWrapper:
    """ Wrapper class for encoding of base layer
    """

    def __init__(self, enc, n_clfs):
        """

        Parameters
        ----------
        enc : Encoding, of size (n_clfs)

        n_clfs : Number of clfs
        """

        self.enc = enc
        self.n_clfs = n_clfs
        self.n_clfs_used = int(np.sum(self.enc))

    def num_of_clfs_used(self):
        """ Get the number of classifiers used in the layer

        Parameters
        ----------
        None

        Returns
        ----------
        n_clfs_used: Scalar
        """

        return self.n_clfs_used

    def clfs_used(self, i_clfs):
        """ Get whether the selected classifier is used in the layer

        Parameters
        ----------
        i_clfs : Classifier index

        Returns
        ----------
        is_used: Scalar (0 or 1)
        """

        return self.enc[i_clfs]

    def print_info(self):
        """ Print the encoding information

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        enc = self.enc
        n_clfs = self.n_clfs
        print('    ------------------------')
        print('    Encoding information:')
        print('    Num of classifiers used: %d' % (n_clfs))
        print('    ', end='', flush=True)
        print(enc)
        print('    ------------------------')

def get_enc_clfs(enc, i_clfs, n_clfs_prev):
    return enc[i_clfs * (n_clfs_prev + 1): (i_clfs + 1) * (n_clfs_prev + 1)]

class LayerEncodingWrapper:
    """ Wrapper class for encoding of ensemble layers (from 2nd onward)
    """

    def __init__(self, enc_in, n_clfs_prev, n_clfs):
        """ Parameters
        ----------
        enc_in : Encoding, of size ((n_clfs_prev+1)*n_clfs)

        n_clfs_prev: The number of classifiers which had output
          in previous layer

        n_clfs : Number of classifiers
        """

        enc = copy.deepcopy(enc_in)
        try:
            enc.ndim
        except:
            enc = np.array(enc, dtype=np.uint8)
  
        self.enc = enc
        self.n_clfs_prev = n_clfs_prev
        self.n_clfs = n_clfs

        n_clfs_out = 0
        clfs_used = np.zeros(n_clfs, dtype=np.uint8)
        for i_clfs in range(n_clfs):
            enc_clfs_i = get_enc_clfs(enc, i_clfs, n_clfs_prev)
            if int(np.sum(enc_clfs_i)) == 0:
                continue
            n_clfs_out += 1
            clfs_used[i_clfs] = 1

        self.n_clfs_out = n_clfs_out
        self.clfs_used = clfs_used

    def get_enc_clfs(self, i_clfs):
        """ Get the sub-encoding related to the i-th classifier

        Parameters
        i_clfs: The index of the classifier
        ----------
        None

        Returns
        ----------
        enc_clfs_i: Numpy array of sub-encoding of i-th classifier, of size
        n_clfs_prev + 1
        """

        enc = self.enc
        n_clfs_prev = self.n_clfs_prev
        n_clfs = self.n_clfs

        enc_clfs_i = get_enc_clfs(enc, i_clfs, n_clfs_prev)
        return copy.deepcopy(enc_clfs_i)

    def num_of_clfs_out(self):
        """ Get the number of classifiers used in the layer

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        return self.n_clfs_out

    def get_clfs_used(self):
        """ Get information about whether each classifier is used or not

        Parameters
        ----------
        None

        Returns
        ----------
        clfs_used: Numpy array, of size (n_clfs), each having value
        of 0 or 1
        """

        return self.clfs_used

    def get_features_used(self, n_cls, n_features):
        """ Get the indices of used features for each classifier
            in the layer

        Parameters
        ----------
        n_cls: Number of classes

        n_features: Number of features

        Returns
        ----------
        features_used: Binary Numpy array of size
        ((n_clfs, n_cls*n_clfs_prev+n_features)), with the first
        n_cls*n_clfs_prev columns for the indices of previous layer,
        and the rest is for the indicies of the original data. Each row indicates the data to be fed
        to the corresponding classifier
        """

        enc = self.enc
        n_clfs_prev = self.n_clfs_prev
        n_clfs = self.n_clfs
        n_clfs_out = self.n_clfs_out
        clfs_used = self.clfs_used

        features_used = np.zeros((n_clfs, n_cls * n_clfs_prev + n_features), dtype=np.uint8)
        for i_clfs in range(n_clfs):
            enc_clfs_i = get_enc_clfs(enc, i_clfs, n_clfs_prev)
            if clfs_used[i_clfs] == 0:
                continue
            if enc_clfs_i[0] == 1:
                # Appends original data
                features_used[i_clfs, n_clfs_prev * n_cls:] = 1
            for j_clfs in range(self.n_clfs_prev):
                if enc_clfs_i[j_clfs + 1] == 1:
                    # Uses metadata from corresponding classifier
                    features_used[i_clfs, (j_clfs * n_cls):((j_clfs + 1) * n_cls)] = 1
        return features_used

    def print_info(self):
        """ Print the encoding information

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """

        n_clfs_prev = self.n_clfs_prev
        n_clfs = self.n_clfs
        enc = self.enc
        print('    ------------------------')
        print('    Encoding information:')
        print('    Num of classifiers used in previous layer: %d' % (n_clfs_prev))
        print('    Num of classifiers used: %d' % (n_clfs))
        for i_clfs in range(n_clfs):
            enc_clfs_i = get_enc_clfs(enc, i_clfs, n_clfs_prev)
            print('')
            print('    Encoding for classifier number %d:' % (i_clfs))
            print('    ', end='', flush=True)
            print(enc_clfs_i)
        print('')
        print('    ------------------------')

def main():
    n_clfs_prev = 2
    n_clfs = 5
    n_features = 4
    n_cls = 3
#    enc = np.zeros((n_clfs_prev+1)*n_clfs, dtype=np.int32
#    )
    enc = np.array([
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
        0, 1, 1,
        0, 0, 0,
        ], dtype=np.int32)
    temp = LayerEncodingWrapper(enc, n_clfs_prev, n_clfs)
    pdb.set_trace()

if __name__ == '__main__':
    main()
