import os
from os import listdir
from os.path import isfile, isdir
import numpy as np
import argparse

## Read desired input from input-file
def load_inputs(input_file, tuples):
    with open(input_file, 'r') as f:
        lines = np.array( f.readlines() )
    start = np.where(lines == '\n')[0] + 1
    if len(start) > 1: start = start[-1]

    parameters = {}
    for tupl in tuples:
        key, typ = tupl[0], tupl[1]
        key_idx = (np.where(lines == f'\t>> {key}\n')[0] - start)[0]
        if typ == bool:
            parameters[key] = typ(int(lines[key_idx]))
        elif typ == str:
            parameters[key] = typ(lines[key_idx])[:-1]
        else:
            parameters[key] = typ(lines[key_idx])
    return parameters


## Create directories
def create_directories(directories):
    if directories[-1] == '/':
        directories = directories[:-1]
    if directories[:2] == './':
        directories = directories[2:]

    path = directories.split('/')
    actual_dir = path.pop(0)
    for idx, new_dir in enumerate(path):
        if idx > 0:
            actual_dir = actual_dir + '/' + path[idx - 1]
        onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
        if (new_dir in onlydirs) == False:
            os.mkdir(f'{actual_dir}/{new_dir}')


## Randomize fraction of a sequence
def randomize_sequence(residues, sequence, fraction):
    assert fraction >= 0. and fraction <= 1., 'Wrong input for fraction variable.'
    if fraction == 0.:
        return sequence
    else:
        mutated_idxs = np.arange(len(sequence))
        if fraction < 1.:
            size = int(fraction * len(sequence))
            idxs = np.arange(len(sequence))
            mutated_idxs = []
            for i in range(size):
                mutated_idx = np.random.randint(len(idxs))
                mutated_idxs.append(idxs[mutated_idx])
                idxs = np.delete(idxs, mutated_idx)
            mutated_idxs = np.sort(mutated_idxs)
        residues = np.array(residues)
        mutant = list(sequence)
        for mutated_idx in mutated_idxs:
            new_residues = residues[residues != sequence[mutated_idx]]
            new_residue = new_residues[np.random.randint(len(new_residues))]
            mutant[mutated_idx] = new_residue
        return ''.join(mutant)


## Load parameters
def load_parameters(d):
    with open(f'{d}/parameters.dat', 'r') as f:
        lines = f.readlines()
    splitted_lines = np.array([line.split('\t') for line in lines])
    parameters = {
            'generation': splitted_lines[:, 0].astype(float),
            'T': splitted_lines[:, 1].astype(float),
            'gamma': splitted_lines[:, 2].astype(float),
            'threshold': splitted_lines[:, 3].astype(float),
            'seed': splitted_lines[:, 4].astype(float),
    }
    return parameters


## Load data
def load_data(d):
    with open(f'{d}/data.dat', 'r') as f:
        lines = f.readlines()
    splitted_lines = np.array([line.split('\t') for line in lines])
    data = {
            'sequences': splitted_lines[:, 1],
            'energies': splitted_lines[:, 2].astype(float),
            'PAM1_distances': splitted_lines[:, 3].astype(float),
            'Hamm_distances': splitted_lines[:, 4].astype(float),
            'ar': splitted_lines[:, 5].astype(float)
    }
    return data


## Load data and parameters
def load_all(d):
    data = load_data(d)
    parameters = load_parameters(d)
    return data, parameters
