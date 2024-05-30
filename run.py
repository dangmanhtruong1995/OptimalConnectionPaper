import os
import pdb
from time import time

import numpy as np

from combiners import SumRuleCombiner
from data_helper import data_folder, file_list
from deep_ensemble import DeepEnsemble
from get_classifiers import get_classifiers
from output_writer import OutputWriter
from encoding_helper import LayerEncodingWrapper

def main():
    config = {}
    config['n_cv_folds'] = 5
    (clfs, clfs_name_list) = get_classifiers()
    n_clfs = len(clfs)

    config['clfs'] = clfs
    config['n_clfs'] = n_clfs
    config['pop_size'] = 50
    config['early_stopping_rounds'] = 3
    config['max_layers'] = 50 # If the number of layers more than this, stop optimize
    
    # config['optimizer_name'] = 'GA' # Binary Genetic Algorithm
    config['optimizer_name'] = 'PSO' # Binary Particle Swarm Optimization
    # config['optimizer_name'] = 'DE' # Binary Differential Evolution
    config['optimizer_early_stopping'] = 10

    # For Binary GA
    config['crossover_prob'] = 0.8
    config['mutation_prob'] = 0.2
    
    # For Binary PSO
    config['c1'] = 0.5
    config['c2'] = 0.5
    config['w'] = 0.9
    config['k'] = 30
    config['p'] = 2
    
    # For Binary DE
    config['DE_F'] = 2 # Scale factor
    config['DE_cr'] = 0.7 # Crossover rate


    print('_____Parameters_____')
    print('nClassifiers = ', n_clfs)
    print('n_cv_folds = ', config['n_cv_folds'])
    print('pop_size = ', config['pop_size'])
    print('early_stopping_rounds = ', config['early_stopping_rounds'])
    print('max_layers = ', config['max_layers'])
    print('crossover_prob = ', config['crossover_prob'])
    print('mutation_prob = ', config['mutation_prob'])
    print('optimizer_name = ', config['optimizer_name'])
    print('optimizer_early_stopping = ', config['optimizer_early_stopping'])
    print('____________________')

    config['meta_clf'] = SumRuleCombiner()

    for file_name in file_list:
        print('Dataset: {}'.format(file_name))

        D_train = np.loadtxt(os.path.join(data_folder, 'train1', file_name + '_train1.dat'), delimiter=',')
        D_val = np.loadtxt(os.path.join(data_folder, 'val', file_name + '_val.dat'), delimiter=',')
        D_test = np.loadtxt(os.path.join(data_folder, 'test', file_name + '_test.dat'), delimiter=',')

        X_train = D_train[:, :-1]
        y_train = D_train[:, -1].astype(np.int32)
        X_val = D_val[:, :-1]
        y_val = D_val[:, -1].astype(np.int32)
        X_test = D_test[:, :-1]
        y_test = D_test[:, -1].astype(np.int32)

        classes = np.unique(np.concatenate((y_train, y_val, y_test)))
        if np.any(classes.astype(np.int32) == 0):
            raise Exception("Labels have to start from 1")

        n_classes = np.size(classes)
        config['n_cls'] = n_classes

        # Run the algorithm
        deep_ensemble = DeepEnsemble(config)
        t1 = time()
        (n_layers, encoding_list, val_acc_list) = deep_ensemble.optimize(
            X_train, y_train, X_val, y_val)
        t2 = time()
        optimize_layers_time = t2 - t1

        X_trainval = np.vstack((X_train, X_val))
        y_trainval = np.concatenate((y_train, y_val))
        t1 = time()
        metric_dict = deep_ensemble.fit_and_test(
            X_trainval, y_trainval, X_test, y_test, n_layers, encoding_list)
        t2 = time()
        run_time = t2 - t1

        # Save the results to file
        output_path = os.path.join(os.getcwd(), 'result', file_name)
        output_writer = OutputWriter(output_path)

        encoding_list_1 = []
        for encoding in encoding_list:
            encoding_list_1.append(encoding.tolist())
        encoding_list = encoding_list_1
        performance_output = {
            "val_acc": val_acc_list,
            "acc": metric_dict["acc"],
            "precision": metric_dict["precision"],
            "recall": metric_dict["recall"],
            "f1": metric_dict["f1"],
            "n_layers": [n_layers + 1],  # Because layer index starts from 0
            "encoding_list": encoding_list,
            "classifier_list": clfs_name_list,
        }
        output_writer.write_output(performance_output, 'performance', indent=2)

        # Save final accuracy, ... to another file
        final_result_output = {
            "Number of optimal layers": n_layers+1 , # Because layer index starts from 0
            "Validation accuracy": val_acc_list[n_layers],
            "Test accuracy": metric_dict['acc'][n_layers],
            "Test F1-score": metric_dict['f1'][n_layers],
            "Time to find optimal layers (in seconds)": optimize_layers_time,
            "Time to run (in seconds)": run_time,
        }
        output_writer.write_output(final_result_output, 'final_result', indent=2)

        # Save parameters used
        parameters_output = {}
        for key, val in config.items():
            if key == 'meta_clf':
                parameters_output[key] = type(val).__name__
            elif key == 'clfs':
                # Classifier list saved in encoding.txt already
                pass
            else:
                parameters_output[key] = val
        output_writer.write_output(parameters_output, 'parameters', indent=2)

        # Encoding file
        with open(os.path.join(output_path, 'encoding.txt'), 'wt') \
                as fid:
            fid.write('List of classifiers:\n')
            for clfs_name in clfs_name_list:
                fid.write('- %s\n' % (clfs_name))
            fid.write('-----------------\n')
            for i_layer, encoding in enumerate(encoding_list):
                if i_layer > n_layers:
                    continue
                fid.write('Layer %d:\n' % (i_layer + 1))
                if i_layer == 0:
                    for enc_pos, enc_val in enumerate(encoding):
                        if int(enc_val) == 1:
                            fid.write('- Train -> %s\n' % (clfs_name_list[enc_pos]))
                else:
                    encoding_size = len(encoding)
                    n_clfs_prev = int(encoding_size / (1.0 * n_clfs)) - 1
                    enc_wrapper = LayerEncodingWrapper(encoding,
                        n_clfs_prev, n_clfs)
                    for i_clfs in range(n_clfs):
                        enc_clfs_i = enc_wrapper.get_enc_clfs(i_clfs)
                        for enc_pos, enc_val in enumerate(enc_clfs_i):
                            if int(enc_val) == 0:
                                continue
                            clfs_name_list[i_clfs]
                            if enc_pos == n_clfs_prev:
                                fid.write('- %s -> %s\n' % (
                                    'Original',
                                    clfs_name_list[i_clfs]))
                            else:
                                fid.write('- %s -> %s\n' % (
                                    clfs_name_list[enc_pos],
                                    clfs_name_list[i_clfs]))

if __name__ == '__main__':
    main()
