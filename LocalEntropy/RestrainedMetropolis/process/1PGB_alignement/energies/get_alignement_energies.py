import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from subprocess import call

from utils import load_inputs
from my_classes import *


def get_distances(wt_sequence, mt_sequence, distmatrix, residues, rescale = True):
    wt_array = np.array(list(wt_sequence))
    mt_array = np.array(list(mt_sequence))
    new_residues_idxs = np.where(wt_array != mt_array)[0]

    # Hamming distance
    Hamm_distance = len(new_residues_idxs)

    # PAM1 distance
    old_residues = wt_array[new_residues_idxs]
    new_residues = mt_array[new_residues_idxs]
    PAM1_distance = 0.
    for old_residue, new_residue in zip(old_residues, new_residues):
        old_idx = residues.index(old_residue)
        new_idx = residues.index(new_residue)
        PAM1_distance += distmatrix[new_idx, old_idx]

    if rescale:
        length = len(wt_sequence)
        PAM1_distance = PAM1_distance/length
        Hamm_distance = Hamm_distance/length
    return PAM1_distance, Hamm_distance


def calculate_effective_energy(wt_contacts, mt_contacts):
    # Modified contacts fraction
    mod_diff = abs(mt_contacts - wt_contacts)
    norm = np.sum(mt_contacts) + np.sum(wt_contacts)
    eff_en = np.sum(mod_diff) / norm
    return eff_en


def main(args):
    print(f'PID: {os.getpid()}')
    print('Loading inputs...')
    tuples = [
        ('input_file', str),
        ('output_file', str),
        ('device', int),
        ('distance_threshold', float)
    ]
    parameters = load_inputs(args.input_file, tuples)
    command = f"rm {parameters['output_file']}".split(' ')
    call(command)

    with open(parameters['input_file'], 'r') as f:
        lines = f.readlines()
    wt_sequence = lines[0].split('\t')[-1][:-1]
    mt_sequences = [line.split('\t')[-1][:-1] for line in lines[1:]]

    distmatrix = pd.read_csv('inputs/DistPAM1.csv')
    distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
    residues = tuple(distmatrix.columns)
    distmatrix = np.array(distmatrix)
    site_weights = np.load('inputs/site_weights.npy')
    

    print('Initiating algorithm...\n')
    algorithm = Basic_class(
            device = parameters['device'],
            distance_threshold = parameters['distance_threshold']
    )
    wt_contacts = algorithm.calculate_contacts(wt_sequence)
    wt_subsequence = ''.join(np.array(list(wt_sequence))[site_weights == 1])

    

    print('Calculating the mutants energies...')
    for imt, mt_sequence in tqdm(enumerate([wt_sequence] + mt_sequences), total = len([wt_sequence] + mt_sequences)):
        if ('-' in mt_sequence) or ('.' in mt_sequence):
            mt_energy, mt_PAM1_distance, mt_Hamm_distance, mt_subsequence, mt_Hamm_subdistance = 'None', 'None', 'None', 'None', 'None'
        
        else:
            mt_contacts = algorithm.calculate_contacts(mt_sequence)
            mt_energy = calculate_effective_energy(wt_contacts, mt_contacts)

            mt_PAM1_distance, mt_Hamm_distance = get_distances(wt_sequence, mt_sequence, distmatrix, residues, rescale = True)
            mt_subsequence = ''.join(np.array(list(mt_sequence))[site_weights == 1])
            _, mt_Hamm_subdistance = get_distances(wt_subsequence, mt_subsequence, distmatrix, residues, rescale = False)

        with open(parameters['output_file'], 'a') as f:
            print(f'{imt}\t{mt_energy}\t{mt_PAM1_distance}\t{mt_Hamm_distance}\t{mt_subsequence}\t{mt_Hamm_subdistance}', file = f)
    print('...done!')



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
