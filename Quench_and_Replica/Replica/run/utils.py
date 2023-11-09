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
            'generation': splitted_lines[:, 0].astype(int),
            'T': splitted_lines[:, 1].astype(float),
            'gamma': splitted_lines[:, 2].astype(float),
            'y': splitted_lines[:, 3].astype(int),
            'seed': splitted_lines[:, 4].astype(int),
    }
    assert len(np.unique(parameters['y'])) == 1, 'Too many saved values for y replica value.'
    return parameters


## Load single replica data
def load_replica(d, replica_idx, y):
    replica_file = np.array([f for f in listdir(d) if isfile(f'{d}/{f}') and (f'replica_{replica_idx}.dat' == f)])
    assert len(replica_file) == 1, 'Too many replica files for the same replica.'
    replica_file = replica_file[0]
    
    with open(f'{d}/{replica_file}', 'r') as f:
        lines = f.readlines()
    splitted_lines = np.array([line.split('\t') for line in lines])
    sequences = splitted_lines[:, 1]
    energies = splitted_lines[:, 2].astype(float)
    Hamm_distances = splitted_lines[:, (3+y):(3+2*y)].astype(float)
    PAM1_distances = splitted_lines[:, 3:(3+y)].astype(float)
    PAM1_distances = splitted_lines[:, 3:(3+y)].astype(float)
    ar = splitted_lines[:, -1].astype(float)
    return sequences, energies, Hamm_distances, PAM1_distances, ar


## Load all replicas data
def load_all(d):
    data = {
        'sequences': np.array([]),
        'energies': np.array([]),
        'Hamm_distances': np.array([]),
        'PAM1_distances': np.array([]),
        'ar' : np.array([]),
    }
    parameters = load_parameters(d)
    y = np.unique(parameters['y'])
    assert len(y) == 1, 'Too many values for y parameter.'
    y = y[0]
    assert len([f for f in listdir(d) if isfile(f'{d}/{f}') and ('replica' in f)]) == y, 'Wrong number of replica files based on simulation parameter y.'
    
    for irep in range(y):
        replica_data = load_replica(d, irep, y)
        replica_data = list(replica_data)
        for i, key in enumerate(data):
            replica_data[i] = np.expand_dims(replica_data[i], axis = 1)
            if irep == 0:
                data[key] = replica_data[i].copy()
            else:
                data[key] = np.append(data[key], replica_data[i], axis = 1)
    return data
