import numpy as np
import argparse
import os
from os import listdir
from os.path import isfile

from integrator import *
from utils import *


def main(args):
    print(f'PID: {os.getpid()}')

    print('Loading parameters...')
    tuples = [
        ('protein_name', str),
        ('T', float),
        ('s', int),
        ('inputs_dir', str),
        ('discarded_mutations', int),
        ('gamma_min', float),
        ('const_step', bool)
    ]
    parameters = load_inputs(args.input_file, tuples)
    
    # MM directory
    dirlist = [d for d in listdir(f'{parameters["inputs_dir"]}') if isdir(f'{parameters["inputs_dir"]}/{d}') and (f'MM_{parameters["protein_name"]}_T{parameters["T"]}' == d)]
    assert len(dirlist) == 1, 'Too many directories.'
    parameters["inputs_dir"] = f'{parameters["inputs_dir"]}/{dirlist[0]}'
    # seed directory
    subdirlist = [d for d in listdir(f'{parameters["inputs_dir"]}') if isdir(f'{parameters["inputs_dir"]}/{d}') and (f's{parameters["s"]}' == d)]
    assert len(subdirlist) == 1, 'Too many directories.'
    parameters["inputs_dir"] = f'{parameters["inputs_dir"]}/{subdirlist[0]}'

    print('Parameters:')
    for key in parameters:
        print(f'- {key}: {parameters[key]}')
    print()
    
    integrator = Integrator(
        inputs_dir = parameters["inputs_dir"],
        discarded_mutations = parameters["discarded_mutations"],
        gamma_min = parameters["gamma_min"],
        const_step = parameters["const_step"],
        initialize = True
    )


    print(f'Protein {parameters["protein_name"]} Local Entropy calculation:')
    print(f'- Simpson method...')
    integrator.Simpson()
    print(f'- Midpoint method...')
    integrator.MidPoint()
    print('Success!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs.txt'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
