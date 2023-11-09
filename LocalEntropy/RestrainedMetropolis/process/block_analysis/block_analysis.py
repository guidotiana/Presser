import argparse
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, isdir

from utils import load_inputs, load_data


def get_progrs(block_mean):
    block_number = len(block_mean)
    progr_mean, progr_err = np.zeros(block_number), np.zeros(block_number)
    for iblock in range(block_number):
        aux = block_mean[:(iblock + 1)].mean()
        aux2 = (block_mean[:(iblock + 1)] ** 2).mean()
        progr_mean[iblock] = aux
        progr_err[iblock] = np.sqrt((aux2 - aux**2)/(iblock + 1))
    return progr_mean, progr_err


def main(args):
    print(f'pid: {os.getpid()}\n')
    print('Loading inputs...')
    tuples = [
        ('protein_name', str),
        ('T', float),
        ('gamma', float),
        ('discarded_mutations', int),
        ('block_size', int),
        ('results_d', str)
    ]
    parameters = load_inputs(args.input_file, tuples)
    parameters["sim_id"] = f'SM_{parameters["protein_name"]}_T{format(parameters["T"], ".4f")}_g{parameters["gamma"]}'

    
    print('Loading data...')
    data = np.array([])
    d = f'../../{parameters["results_d"]}/{parameters["sim_id"]}'
    seed_ds = [subd for subd in listdir(f'{d}') if isdir(f'{d}/{subd}')]
    for seed_d in seed_ds:
        actual_d = f'{d}/{seed_d}'
        seed_data = load_data(actual_d, return_eq = False)
        seed_data = seed_data[parameters["discarded_mutations"]:]
        seed_data[:, 3] = seed_data[:, 3] / seed_data[0, -1]
        seed_data[:, 4] = seed_data[:, 4] / seed_data[0, -1]
        if len(data) == 0: data = seed_data.copy() 
        else: data = np.append(data, seed_data, axis = 0)
    parameters["discarded_mutations"] = parameters["discarded_mutations"] * len(seed_ds)
    parameters["length"] = seed_data[0, -1]

    print('Parameters:')
    for key in parameters:
        print(f'{key}: {parameters[key]}')

    
    print('\nPerforming block analysis...')
    tuples = [
            ('energy', 1, 1./parameters["T"]),
            ('PAM1 distance', 3, parameters["gamma"]),
            ('Hamm distance', 4, parameters["gamma"])
    ]

    block_data = {}
    for tupl in tuples:
        (name, idx, par) = tupl
        ob = data[:, idx]
        block_number = int( len(ob)/parameters["block_size"] )
        ob_block_mean, ob2_block_mean = np.zeros(block_number), np.zeros(block_number)
        
        for iblock in range(block_number):
            ob_block_mean[iblock] = ob[(iblock*parameters["block_size"]) : ((iblock+1)*parameters["block_size"])].mean()
            ob2_block_mean[iblock] = (ob[(iblock*parameters["block_size"]) : ((iblock+1)*parameters["block_size"])] ** 2.).mean()
        
        ## mean
        ob_progr_mean, ob_progr_err = get_progrs(ob_block_mean)
        
        ## specific heat
        ob2_progr_mean, ob2_progr_err = get_progrs(ob2_block_mean)
        C_progr_mean = (par ** 2.) * (ob2_progr_mean - (ob_progr_mean ** 2.))
        C_progr_err = abs(par) * np.sqrt( ob2_progr_err + 2.*ob_progr_mean*ob_progr_err )

        block_data[f'{name}'] = ob_progr_mean
        block_data[f'{name} error'] = ob_progr_err
        block_data[f'{name} chi'] = C_progr_mean
        block_data[f'{name} chi error'] = C_progr_err

    block_data['block'] = np.arange(block_number) + 1
    block_data['block_size'] = [parameters["block_size"]] * block_number
    block_data['used_mutations'] = [parameters["block_size"] * block_number] * block_number
    block_data['discarded_mutations'] = [parameters["discarded_mutations"]] * block_number
    block_data['T'] = [parameters["T"]] * block_number
    block_data['gamma'] = [parameters["gamma"]] * block_number
    block_data['length'] = [parameters["length"]] * block_number
    

    print('\nSaving block data...')
    filepath = f'../../{parameters["results_d"]}/{parameters["sim_id"]}'
    filename = f'block_data_T{parameters["T"]}_g{parameters["gamma"]}'
    block_data = pd.DataFrame(block_data)
    block_data.to_csv(f'{filepath}/{filename}.csv')



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
