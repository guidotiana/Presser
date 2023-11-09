import os
import argparse
import numpy as np
from time import time
from os import listdir
from os.path import isdir
from subprocess import call

from utils import load_inputs, create_directories
from replica_classes import *


def check_parameters(parameters):
    # Check for <protein_name> directory
    results_dir = f"{parameters['results_dir']}/{parameters['protein_name']}/y{parameters['y']}_s{parameters['seed']}_T{parameters['T']}"
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
        ('y', int),
        ('seed', int),
        ('T', float),
        ('gmfile', str),
        ('unique_length', int),
        ('results_dir', str),
        ('restart', bool),
        ('step', int),
        ('device', int),
        ('distance_threshold', float)
    ]
    parameters = load_inputs(args.input_file, tuples)
    parameters = check_parameters(parameters)

    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = parameters['wt_sequence'],
            T = parameters['T'],
            y = parameters['y'],
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
    algorithm.metropolis(parameters['gmfile'])
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
