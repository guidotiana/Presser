import os
import argparse
import numpy as np
from time import time

from utils import load_inputs
from my_classes import *


def generate_random_sequence(wt_sequence):
    distmatrix = pd.read_csv('inputs/DistPAM1.csv')
    distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
    residues = tuple(distmatrix.columns)
    
    indexes = np.random.randint(len(residues), size = len(wt_sequence))
    mutant = ''.join([residues[index] for index in indexes])
    if mutant == wt_sequence:
        mutant = generate_random_sequence(wt_sequence)
        return mutant
    else:
        return mutant


def check_parameters(parameters):
    # Define gammas
    parameters['gamma_i'], parameters['gamma_f'] = np.min([parameters['gamma_i'], parameters['gamma_f']]), np.max([parameters['gamma_i'], parameters['gamma_f']])
    parameters['gammas'] = np.linspace(parameters['gamma_i'], parameters['gamma_f'], parameters['num_gammas'])[::-1]

    # Sequence check
    assert np.any([not parameters['casual_ref'], not parameters['casual_start']]), 'There is no sense in using both reference and starting sequence as casual ones. Choose only one of them.'
    if parameters['ref_sequence'] == '': 
        if parameters['casual_ref']: parameters['ref_sequence'] = generate_random_sequence(parameters['wt_sequence'])
        else: parameters['ref_sequence'] = parameters['wt_sequence']
    
    if parameters['starting_sequence'] == '': 
        if parameters['casual_start']: parameters['starting_sequence'] = generate_random_sequence(parameters['wt_sequence'])
        else: parameters['starting_sequence'] = parameters['ref_sequence']
    
    # Simtype check
    assert parameters['simtype'] in ('SM', 'MM'), 'Wrong simulation type input.'
    if parameters['simtype'] == 'MM' and len(parameters['gammas']) == 1:
        parameters['simtype'] = 'SM'
        print('Since only one gamma was passed, switching simtype from "MM" to "SM".')
    if parameters['simtype'] == 'MM' and parameters['restart_bool']:
        parameters['restart_bool'] = False
        print('Restart option is not available for MM simulations. Setting restart_bool = False.')

    if parameters['simtype'] == 'SM':
        assert len(parameters['gammas']) == 1, 'Too many gamma according to the simulation type.'
        parameters['results_dir'] = f"{parameters['results_dir']}/{parameters['simtype']}_{parameters['protein_name']}_T{str(parameters['T'])}_g{str(parameters['gamma_i'])}/s{parameters['seed']}"
    if parameters['simtype'] == 'MM':
        parameters['results_dir'] = f"{parameters['results_dir']}/{parameters['simtype']}_{parameters['protein_name']}_T{str(parameters['T'])}/s{parameters['seed']}"

    return parameters


def main(args):
    print('Defining wild-type sequence and simulation parameters...')
    tuples = [
        ('simtype', str),
        ('protein_name', str),
        ('wt_sequence', str),
        ('ref_sequence', str),
        ('starting_sequence', str),
        ('casual_ref', bool),
        ('casual_start', bool),
        ('metr_mutations', int),
        ('eq_mutations', int),
        ('T', float),
        ('gamma_i', float),
        ('gamma_f', float),
        ('num_gammas', int),
        ('seed', int),
        ('unique_length', int),
        ('results_dir', str),
        ('restart_bool', bool),
        ('device', int),
        ('distance_threshold', float)
    ]
    parameters = load_inputs(args.input_file, tuples)
    parameters = check_parameters(parameters)

    if parameters['casual_ref']: print(f'Casual reference sequence: {parameters["casual_ref"]}')
    elif parameters['casual_start']: print(f'Casual starting sequence: {parameters["casual_start"]}')

    print('Initiating algorithm...\n')
    algorithm = Mutation_class(
            wt_sequence = parameters['wt_sequence'],
            ref_sequence = parameters['ref_sequence'],
            starting_sequence = parameters['starting_sequence'],
            metr_mutations = parameters['metr_mutations'],
            eq_mutations = parameters['eq_mutations'],
            T = parameters['T'],
            seed = parameters['seed'],
            unique_length = parameters['unique_length'],
            results_dir = parameters['results_dir'],
            restart_bool = parameters['restart_bool'],
            device = parameters['device'],
            distance_threshold = parameters['distance_threshold']
    )
    
    print('Starting the simulation...')
    for igamma, gamma in enumerate(parameters['gammas']):
        algorithm.set_gamma(gamma)
        algorithm.set_starting_sequence(parameters['starting_sequence'])
        algorithm.set_restart_bool(parameters['restart_bool'])
        
        print(f'- Simulation gamma: {algorithm.get_gamma()}')
        if igamma == 0 and parameters['eq_mutations'] > 0:
            print(f'  Equilibration phase...')
            algorithm.metropolis(equilibration = True)
        print('  Metropolis...')
        algorithm.metropolis()
        
        parameters['starting_sequence'] = algorithm.get_last_sequence()
    print('Done!')


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
