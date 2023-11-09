import os
from os import listdir
from os.path import isdir, isfile
import argparse
import numpy as np
from tqdm import tqdm

from utils import *
from my_classes import *


def main(args):
    print('Loading parameters and esmfold model...')
    tuples = [
        ('results_dir', str),
        ('mt_sims', bool),
        ('mt_num', int),
        ('device', int),
        ('distance_threshold', float)
    ]
    parameters = load_inputs(args.input_file, tuples)
    assert parameters["mt_num"] > 0, 'Invalid value for "mt_num" parameter. Valid range: mt_num > 0.'
    algorithm = Basic_class(
            parameters["device"],
            parameters["distance_threshold"]
    )


    print('Loading and sorting directories names...')
    if parameters["mt_sims"]: dirlist = [d for d in listdir(parameters["results_dir"]) if isdir(f'{parameters["results_dir"]}/{d}') and ('g0.0' in d)]
    else: dirlist = [d for d in listdir(parameters["results_dir"]) if isdir(f'{parameters["results_dir"]}/{d}') and ('g0.0' in d) and (not 'mt' in d)]
    
    Tlist = [float(d.split('_')[-2][1:]) for d in dirlist]
    sorted_Tlist = np.sort(Tlist)
    sorted_dirlist = [dirlist[Tlist.index(T)] for T in sorted_Tlist]
    

    print('Calculating plddt profile...')
    for idx, (d, T) in enumerate(zip(sorted_dirlist, sorted_Tlist)):
        print(f'- {d} directory...')

        block_file = [f for f in listdir(f'{parameters["results_dir"]}/{d}') if isfile(f'{parameters["results_dir"]}/{d}/{f}') and ('block_data' in f)]
        assert len(block_file) == 1, 'Block data file not found'
        block_file = block_file[0]
        block_data = pd.read_csv(f'{parameters["results_dir"]}/{d}/{block_file}')

        seed_ds = [subd for subd in listdir(f'{parameters["results_dir"]}/{d}') if isdir(f'{parameters["results_dir"]}/{d}/{subd}')]
        discarded_mutations = int(block_data.loc[0, 'discarded_mutations'] / len(seed_ds))
        mutants = np.array([])
        for seed_d in seed_ds:
            actual_d = f'{parameters["results_dir"]}/{d}/{seed_d}'
            muts = load_muts(actual_d, return_eq = False, return_pars = False)
            if len(mutants) == 0: mutants = muts[discarded_mutations:, 1].copy()
            else: mutants = np.append(mutants, muts[discarded_mutations:, 1], axis = 0)
        
        indexes = np.random.randint(len(mutants), size = parameters["mt_num"])
        indexed_mutants = mutants[indexes]

        mutants_plddt = np.zeros(len(indexes))
        mutants_plddt_err2 = np.zeros(len(indexes))
        for imut, mutant in enumerate(tqdm(indexed_mutants, total = len(indexed_mutants), desc = 'plddt calculation')):
            _, Cas_plddt = algorithm.calculate_contacts(mutant, method = 'explicit', return_plddt = True)
            mutants_plddt[imut] = Cas_plddt.mean()
            mutants_plddt_err2[imut] = Cas_plddt.var()

        if idx == 0: mod = 'w'
        else: mod = 'a'
        with open(args.output_file, mod) as f:
            print(f'{T}\t{format(mutants_plddt.mean(), ".10f")}\t{format(np.sqrt(mutants_plddt_err2.mean()), ".10f")}', file = f)
        print('  done!')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type = str,
        default = "inputs.txt",
        help = "str variable, input file used to load parameters values. Default: 'inputs.txt'."
    )
    parser.add_argument(
        "--output-file",
        type = str,
        default = "plddt_profile.txt",       
        help = "str variable, output file used to save plddt profile data. Default: 'plddt_profile.txt'."       
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
