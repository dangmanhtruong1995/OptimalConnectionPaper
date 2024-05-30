import copy
import pdb
import numpy as np
import time
from pdb import set_trace
# Import the optimization libraries
from bga import BGA
import pyswarms as ps
# from scipy.optimize import differential_evolution
from differential_evolution import differential_evolution

# Import the custom libraries
from base_layer import BaseLayer
from encoding_helper import BaseEncodingWrapper, LayerEncodingWrapper
from ensemble_layer import EnsembleLayer
from utils import collect_result

# BINARY_PSO_OPTIONS = {'c1': 0.5, 'c2': 0.5, 'w':0.9, 'k': 30, 'p':2}

def get_binary_pso_objective_function_all_candidates(func):
    def binary_pso_objective_function_all_candidates(x):
        """Higher-level method to do classification in the 
        whole swarm using Binary PSO.
        Note that the algorithm solves a minimization problem.

        Inputs
        ------
        x: numpy.ndarray of shape (n_particles, dimensions)
            The swarm that will perform the search

        Returns
        -------
        numpy.ndarray of shape (n_particles, )
            The computed objective value for each particle
        """
        n_particles = x.shape[0]
        j = [func(x[i]) for i in range(n_particles)]
        
        return np.array(j)

    return binary_pso_objective_function_all_candidates

class OptimizerWrapper:
    """ Wrapper for the optimization algorithms
    """
    def __init__(self, optimizer_name, pop_size, n_dim, obj_func, config):
        """ Initialization

        Parameters
        ----------
        optimizer_name: Name of the optimization algorithm

        pop_size: Population size

        n_dim: Number of dimensions

        obj_func: The objective function (maximization)
        
        config: A dictionary storing the config options

        Returns
        -------
        None

        """
        
        self.config = config
        
        if optimizer_name == "GA":
            optimizer = BGA(
                pop_shape=(pop_size, n_dim),
                method=obj_func,
                p_c=config["crossover_prob"],
                p_m=config["mutation_prob"],
                early_stop_rounds=config["optimizer_early_stopping"],
                verbose=None,
                maximum=True,
            )            
        elif optimizer_name == "PSO":
            BINARY_PSO_OPTIONS = {}
            BINARY_PSO_OPTIONS['c1'] = config['c1']
            BINARY_PSO_OPTIONS['c2'] = config['c2']
            BINARY_PSO_OPTIONS['w'] = config['w']
            BINARY_PSO_OPTIONS['k'] = config['k']
            BINARY_PSO_OPTIONS['p'] = config['p']
        
            optimizer = ps.discrete.BinaryPSO(
                n_particles=pop_size,
                dimensions=n_dim,
                ftol=0.01,
                ftol_iter=config["optimizer_early_stopping"],
                options=BINARY_PSO_OPTIONS)            
            obj_func_minimize = lambda x : 1-obj_func(x) # For Binary PSO, the library performs a minimization problem
            obj_func_all_candidates = get_binary_pso_objective_function_all_candidates(obj_func_minimize)
            self.obj_func_all_candidates = obj_func_all_candidates            
        elif optimizer_name == "DE":
            bounds = []
            for dim_idx in range(n_dim):
                bounds.append((0, 1))
            bounds = np.array(bounds)
            
            # Here we convert float to binary
            obj_func_minimize = lambda x : 1-obj_func(np.array([round(elem) for elem in x])) # For Binary DE, the library performs a minimization problem
            
            optimizer = differential_evolution 
            
            self.bounds = bounds
            # self.integrality = integrality
            self.obj_func_minimize = obj_func_minimize  
            self.pop_size = pop_size
            self.optimizer_early_stopping = config["optimizer_early_stopping"]

        else:
            raise Error("Optimization algorithm: %s not supported" % (optimizer_name))           
        
        self.optimizer_name = optimizer_name
        self.optimizer = optimizer
        
    def run(self):
        optimizer_name = self.optimizer_name
        optimizer = self.optimizer        
    
        if optimizer_name == "GA":
            print("Running GA!")
            (enc, obj_value) = optimizer.run()            
        elif optimizer_name == "PSO":
            obj_func_all_candidates = self.obj_func_all_candidates
            
            print("Running PSO!")
            cost, enc = optimizer.optimize(
                obj_func_all_candidates,
                iters=1000,
                verbose=2)
            obj_value = 1 - cost # From error to accuracy        
        elif optimizer_name == "DE":
            bounds = self.bounds           
            obj_func_minimize = self.obj_func_minimize
            pop_size = self.pop_size            
            optimizer_early_stopping = self.optimizer_early_stopping    
            config = self.config
            
            F = config['DE_F']
            cr = config['DE_cr']
            
            iter = 1000
            solution = differential_evolution(obj_func_minimize, pop_size, bounds, iter, F, cr, early_stopping=optimizer_early_stopping)
            [enc, cost, _] = solution
            enc = np.array(enc)
            enc = np.array([round(elem) for elem in enc])
            
            obj_value = 1 - cost # From error to accuracy             
            # set_trace()
        return (enc, obj_value)
        
        
class DeepEnsemble:
    """ Class for the deep ensemble evolutionary connection model
    """

    def __init__(self, config):
        self.config = config
        self.meta_clf = config['meta_clf']
        self.clfs = config['clfs']
        self.n_cv_folds = config['n_cv_folds']
        self.n_cls = config['n_cls']
        self.n_clfs = config['n_clfs']
        self.early_stopping_rounds = config['early_stopping_rounds']
        self.max_layers = config['max_layers']

        # For the optimization algorithm
        self.pop_size = config['pop_size']
        self.optimizer_early_stopping = config['optimizer_early_stopping']
        self.crossover_prob = config['crossover_prob'] # For Binary GA only
        self.mutation_prob = config['mutation_prob'] # For Binary GA only
        self.optimizer_name = config['optimizer_name']
        

    def optimize(self, X_train, y_train, X_val, y_val):
        """ Find the optimal encoding for each layer

        Parameters
        ----------
        X_train: Train data, of size (n_train_inst, n_features)

        y_train: Train labels, of size (n_train_inst)

        X_val: Validation data, of size (n_val_inst, n_features)

        y_val: Validation labels, of size (n_val_inst)

        Returns
        -------
        best_n_layers: Optimal number of layers

        enc_list: List of encodings for each layer

        acc_list: List of accuracy for each layer

        """

        config = self.config
        meta_clf = self.meta_clf
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_features = X_train.shape[1]
        early_stopping_rounds = self.early_stopping_rounds
        max_layers = self.max_layers
        pop_size = self.pop_size
        optimizer_early_stopping = self.optimizer_early_stopping
        crossover_prob = self.crossover_prob
        mutation_prob = self.mutation_prob
        optimizer_name = self.optimizer_name

        enc_list = []
        acc_list = []
        layer_idx = 0
        n_layers = 0
        best_acc = -1
        best_n_layers = -1
        X_train_in = copy.deepcopy(X_train)
        X_val_in = copy.deepcopy(X_val)

        print("Start searching...")

        while True:
            # Find the optimal encoding
            print('------------------------')
            print('Optimizing layer {}'.format(layer_idx))
            if layer_idx == 0:
                # First, precompute metadata
                first_layer = BaseLayer(
                    copy.deepcopy(clfs),
                    n_cv_folds,
                    n_cls,
                    n_clfs,
                    meta_clf)
                enc_all = np.ones(n_clfs, dtype=np.uint8)
                (precompute_metadata_train, precompute_metadata_val, _) = \
                    first_layer.run(X_train_in, y_train, X_val_in, y_val, enc_all)

                # Then, optimize
                first_layer = BaseLayer(
                    copy.deepcopy(clfs),
                    n_cv_folds,
                    n_cls,
                    n_clfs,
                    meta_clf,
                    precompute_metadata_train=precompute_metadata_train,
                    precompute_metadata_val=precompute_metadata_val,
                    )
                n_dim_current_layer = n_clfs
                
                t1 = time.time()                
                obj_func = lambda x: first_layer.fitness(
                    X_train_in, y_train, X_val_in, y_val, x)
                optimizer = OptimizerWrapper(optimizer_name, pop_size, n_dim_current_layer, obj_func, config)
                enc, acc = optimizer.run() 
                # set_trace()
                t2 = time.time()                
                print('Running current layer took: %f seconds.' % (t2-t1))
                
                if int(np.sum(enc)) == 0:
                    enc[enc == 0] = 1
                enc_wrapper = BaseEncodingWrapper(enc, n_clfs)
                # set_trace()
                (X_train_in, X_val_in, _) = first_layer.run(
                    X_train_in, y_train, X_val_in, y_val, enc)
                n_clfs_prev = enc_wrapper.num_of_clfs_used()
            else:
                current_layer = EnsembleLayer(
                    copy.deepcopy(clfs),
                    n_cv_folds,
                    n_cls,
                    n_clfs,
                    n_clfs_prev,
                    meta_clf)
                n_dim_current_layer = n_clfs * (n_clfs_prev + 1)
                
                t1 = time.time()                
                obj_func = lambda x: current_layer.fitness(
                    X_train_in, y_train, X_val_in, y_val, X_train, X_val, x)
                optimizer = OptimizerWrapper(optimizer_name, pop_size, n_dim_current_layer, obj_func, config)
                enc, acc = optimizer.run()
                t2 = time.time()
                print('Running current layer took: %f seconds.' % (t2-t1))
                
                if int(np.sum(enc)) == 0:
                    enc[enc == 0] = 1
                enc_wrapper = LayerEncodingWrapper(enc,
                                                   n_clfs_prev, n_clfs)
                (X_train_in, X_val_in, _) = current_layer.run(
                    X_train_in, y_train, X_val_in, y_val, X_train, X_val, enc)
                n_clfs_prev = enc_wrapper.num_of_clfs_out()

            acc_list.append(acc)
            enc_list.append(enc)
            print("Finish optimizing for current layer!")
            enc_wrapper.print_info()
            print("Fitness:")
            print(acc)
            if best_acc < acc:
                best_acc = acc
                best_n_layers = layer_idx
            else:
                if layer_idx - best_n_layers >= early_stopping_rounds:
                    break
            layer_idx += 1
            if (best_n_layers+1) > max_layers:
                break

        return (best_n_layers, enc_list, acc_list)

    def fit_and_test(self, X_trainval, y_trainval, X_test, y_test,
                     n_layers, enc_list):
        """ Fit and test in one function so that we don't have to
            store all intermediate layers

        Parameters
        ----------
        X_trainval: Trainval data, of size (n_trainval_inst, n_features)

        y_trainval: Trainval labels, of size (n_trainval_inst)

        X_test: Test data, of size (n_test_inst, n_features)

        y_test: Test labels, of size (n_test_inst)

        n_layers: Number of optimal layers

        enc_list: List of encoding (list of list)

        Returns
        -------
        metric_dict: A dictionary containing metrics for test result
        on each layer
        """
        meta_clf = self.meta_clf
        clfs = self.clfs
        n_cv_folds = self.n_cv_folds
        n_cls = self.n_cls
        n_clfs = self.n_clfs
        n_features = X_trainval.shape[1]

        metric_dict = {}
        metric_dict['acc'] = []
        metric_dict['precision'] = []
        metric_dict['recall'] = []
        metric_dict['f1'] = []

        layer_idx = 1
        n_layers = 0
        best_acc = -1
        best_n_layers = -1

        X_trainval_in = copy.deepcopy(X_trainval)
        X_test_in = copy.deepcopy(X_test)

        print("Test after optimize!")

        for layer_idx in range(len(enc_list)):
            enc = enc_list[layer_idx]
            if isinstance(enc, list):
                enc = np.array(enc, dtype=np.int32)
            if layer_idx == 0:
                # First layer
                first_layer = BaseLayer(clfs, n_cv_folds, n_cls, n_clfs, meta_clf)
                (X_trainval_in, X_test_in, pred_test) = first_layer.run(
                    X_trainval_in, y_trainval, X_test_in, y_test, enc)
                enc_wrapper = BaseEncodingWrapper(enc, n_clfs)
                n_clfs_prev = enc_wrapper.num_of_clfs_used()
            else:
                # From 2nd layer onward
                current_layer = EnsembleLayer(clfs, n_cv_folds, n_cls,
                                              n_clfs, n_clfs_prev, meta_clf)
                (X_trainval_in, X_test_in, pred_test) = current_layer.run(
                    X_trainval_in, y_trainval, X_test_in, y_test, X_trainval, X_test, enc)

                enc_wrapper = LayerEncodingWrapper(enc,
                                                   n_clfs_prev, n_clfs)
                n_clfs_prev = enc_wrapper.num_of_clfs_out()

            result = collect_result(y_test, pred_test)
            for metric in metric_dict:
                metric_dict[metric].append(result[metric])
            acc = result['acc']

            print("")
            print("Layer: %d" % (layer_idx))
            enc_wrapper.print_info()
            print("Fitness:")
            print(acc)

        return metric_dict
