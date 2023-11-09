import os
import argparse
import numpy as np
from time import time
from os import listdir
from os.path import isdir, isfile
from subprocess import call

from utils import load_inputs, create_directories
from quench_classes import *


def check_parameters(parameters):
    # Check for <protein_name> directory and save <protein_name> data
    results_dir = f"{parameters['results_dir']}/{parameters['protein_name']}/s{parameters['seed']}"
    create_directories(results_dir)

    first_dir = f"{parameters['results_dir']}/{parameters['protein_name']}"
    saved_files = [f for f in listdir(parameters['results_dir']) if isfile(f"{parameters['results_dir']}/{f}")]
    if not 'README.txt' in saved_files:
        with open(f"{first_dir}/README.txt", 'w') as f:
            print(f"Protein name:     {parameters['protein_name']}", file = f)
            print(f"Protein sequence: {parameters['wt_sequence']}", file = f)
            print(f"Protein length:   {len(parameters['wt_sequence'])}", file = f)
    
    parameters['results_dir'] = results_dir
    
    return parameters


def main(args):
    print(f'PID: {os.getpid()}\n')

    print('Defining wild-type sequence and simulation parameters...')
    tuples = [
        ('protein_name', str),
        ('wt_sequence', str),
        ('ref_sequence', str),
        ('Tgt_file', str),
        ('seed', int),
        ('unique_length', int),
        ('results_dir', str),
        ('restart', bool),
        ('step', int),
        ('device', int),
        ('distance_threshold', float)
    ]
    parameters = load_inputs(args.input_file, tuples)
    parameters = check_parameters(parameters)

    with open(parameters['Tgt_file'], 'r') as f:
        lines = f.readlines()
    assert len(lines) == 1, f'Too many lines for quench algorithm. Expected lines 1, Found lines {len(lines)}.'
    lines = lines[0].split('\t')
    if len(lines) == 2:
        T, gamma = float(lines[0]), float(lines[1])
        df = pd.read_csv('inputs/processed_data.csv')
        threshold = df.loc[df.loc[:, 'T'] == T, 'energy'].iloc[0]
        line = f'{T}\t{gamma}\t{threshold}'
        with open(parameters['Tgt_file'], 'w') as f:
            print(line, file = f)
    else:
        if len(lines) != 3: raise ValueError(f'Wrong number of element in {parameters["Tgt_file"]}. Expected 2 or 3, Found {len(lines)}.')

    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = parameters['wt_sequence'],
            ref_sequence = parameters['ref_sequence'],
            seed = parameters['seed'],
            unique_length = parameters['unique_length'],
            results_dir = parameters['results_dir'],
            step = parameters['step'],
            restart = parameters['restart'],
            device = parameters['device'],
            distance_threshold = parameters['distance_threshold']
    )
    
    headline = ''.join(['-'] * 100)
    print(f'{headline}\nStarting the simulation...')
    algorithm.metropolis(parameters['Tgt_file'])
    print('Simulation completed!')
    

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs/inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs/inputs.txt'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
