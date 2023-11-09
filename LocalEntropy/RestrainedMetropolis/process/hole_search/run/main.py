import argparse
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, isdir

from utils import load_inputs, load_muts
from distribution_class import *


def load_and_save(algorithm, parameters, mutants):
    if parameters["wtc"] == "cd":
        print('\nStarting the calculation for characteristic dimension...')
        if parameters["ref_mutant"] == '': parameters["ref_mutant"] = mutants[0]
        results = algorithm.calculate_cd(mutants, parameters["ref_mutant"], return_pdf = True)

    elif parameters["wtc"] == "pdf":
        print('\nStarting the calculation for pdf...')
        results = algorithm.calculate_pdf(mutants)

    else:
        print('\nStarting the sanity check...')
        total_size = len(mutants)
        group_size = int(total_size / parameters["groups_number"])

        results = {}
        for igroup in range(parameters["groups_number"]):
            print(f'< {(igroup + 1) * group_size} group:')
            group_mutants = mutants[:(igroup + 1) * group_size]
            group_results = algorithm.calculate_pdf(group_mutants)
            if igroup == 0:
                results['distances'] = group_results['distances']

            if igroup < parameters["groups_number"] - 1:
                results[f'< {(igroup + 1) * group_size}'] = group_results['pdf']
            else:
                results['total'] = group_results['pdf']
        results = pd.DataFrame(results)

    print('Saving results...')
    filename = f'{parameters["wtc"]}_{parameters["dtype"]}_p{parameters["prec"]}_s{parameters["step"]}'
    if parameters["weighted"]: filename = f'weighted_{filename}'
    else: filename = f'simple_{filename}'
    results.to_csv(f'{parameters["results_d"]}/{parameters["sim_id"]}/{filename}.csv')
    print('Done!')


def main(args):
    print(f'pid: {os.getpid()}\n')
    print('Loading inputs...')
    tuples = [
        ('protein_name', str),
        ('T', float),
        ('gamma', float),
        ('dtype', str),
        ('prec', float),
        ('step', int),
        ('def_disc_mut', int),
        ('weighted', bool),
        ('inputs_d', str),
        ('results_d', str),
        ('wtc', str),
        ('ref_mutant', str),
        ('groups_number', int)
    ]
    parameters = load_inputs(args.input_file, tuples)
    assert parameters["wtc"] in ('all', 'cd', 'pdf', 'sc'), 'Wrong input for wtc variable. Allowed values: "all", "cd", "pdf" and "sc".'
    parameters["sim_id"] = f'SM_{parameters["protein_name"]}_T{parameters["T"]}_g{parameters["gamma"]}'
    assert parameters["groups_number"] >= 0, 'Wrong input for groups_number variable. Allowed values: >= 0.'
    if parameters["groups_number"] == 0: parameters["groups_number"] = 5
    if parameters["dtype"] == 'Hamm' and not parameters["weighted"]: parameters["prec"] = 1.
    assert parameters["step"] > 0, 'Wrong input for step variable. Allowed values: > 0.'
    
    
    print('Loading mutants...')
    check_sim = [(parameters["sim_id"] in d) for d in listdir(parameters["results_d"]) if isdir(f'{parameters["results_d"]}/{d}')]
    if not np.any(check_sim): os.mkdir(f'{parameters["results_d"]}/{parameters["sim_id"]}')
    
    seed_ds = [subd for subd in listdir(f'../../{parameters["results_d"]}/{parameters["sim_id"]}') if isdir(f'../../{parameters["results_d"]}/{parameters["sim_id"]}/{subd}')]
    block_file = [f for f in listdir(f'../../{parameters["results_d"]}/{parameters["sim_id"]}') if isfile(f'../../{parameters["results_d"]}/{parameters["sim_id"]}/{f}') and ('block_data' in f)]
    if len(block_file) == 0:
        parameters["discarded_mutations"] = parameters["def_disc_mut"]
        print(f'No block data file found. Performing calculation with default discarded mutations value : {parameters["discarded_mutations"]}')
    else:
        assert len(block_file) == 1, 'Too many block_data files.'
        block_file = block_file[0]
        block_data = pd.read_csv(f'../../{parameters["results_d"]}/{parameters["sim_id"]}/{block_file}')
        parameters["discarded_mutations"] = int(block_data.loc[0, 'discarded_mutations'] / len(seed_ds))
    
    mutants = np.array([])
    for seed_d in seed_ds:
        actual_d = f'../../{parameters["results_d"]}/{parameters["sim_id"]}/{seed_d}'
        seed_muts = load_muts(actual_d, return_eq = False, return_pars = False)
        seed_mutants = seed_muts[:, 1].astype(str)
        if len(mutants) == 0: mutants = seed_mutants[parameters["discarded_mutations"]:].copy()
        else: mutants = np.append(mutants, seed_mutants[parameters["discarded_mutations"]:], axis = 0)
    mutants = mutants[::parameters["step"]]

    print('Preparing algorithm...')
    algorithm = Distribution_class(
            dtype = parameters["dtype"],
            prec = parameters["prec"],
            weighted = parameters["weighted"],
            inputs_d = parameters["inputs_d"]
    )
    print('\nProgram parameters:')
    for key in parameters: print(f'- {key}: {parameters[key]}')


    if parameters["wtc"] != "all":
        load_and_save(algorithm, parameters, mutants)
    else:
        for wtc in ("cd", "pdf", "sc"):
            parameters["wtc"] = wtc
            load_and_save(algorithm, parameters, mutants)



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
