import argparse
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, isdir

from utils import load_data


def main(args):    
    print('Ordering directories...')
    dirlist = [d for d in listdir(f'{args.results_dir}') if isdir(f'{args.results_dir}/{d}') and ('SM' in d) and (not '.ipynb' in d)]
    T_arr, gamma_arr, d_arr = np.array([]), np.array([]), np.array([])

    for d in dirlist:
        check = np.any(['block_data' in f for f in listdir(f'{args.results_dir}/{d}') if isfile(f'{args.results_dir}/{d}/{f}')])
        if not check:
            continue
        else:
            T = float(d.split('_')[2][1:])
            gamma = float(d.split('_')[3][1:])
                
            T_arr = np.append(T_arr, [T], axis = 0)
            gamma_arr = np.append(gamma_arr, [gamma], axis = 0)
            d_arr = np.append(d_arr, [f'{args.results_dir}/{d}'], axis = 0)

    sorted_T_arr = np.sort(T_arr)
    dirlist = np.array([])
    for T in np.unique(sorted_T_arr):
        mask = T_arr == T
        masked_gamma_arr = gamma_arr[mask]
        masked_d_arr = d_arr[mask]
        
        sorted_masked_gamma_arr = np.sort(masked_gamma_arr)
        for gamma in np.unique(sorted_masked_gamma_arr):
            mask = masked_gamma_arr == gamma
            dirlist = np.append(dirlist, masked_d_arr[mask], axis = 0)

    print('Processing data...')
    columns_check = False
    data_array = np.array([])
    for d in dirlist:
        print(f'- d: {d.split("/")[-1]}')
        csvfile = [f for f in listdir(f'{d}') if isfile(f'{d}/{f}') and ('block_data' in f)]
        assert len(csvfile) == 1, 'Too many block data files.'
        csvfile = csvfile[0]
        block_data = pd.read_csv(f'{d}/{csvfile}')
        block_data = block_data.drop(columns = ['Unnamed: 0', 'block'])

        if not columns_check:
            columns = list(block_data.columns) 
            columns_check = True

        new_data = np.array(block_data.loc[len(block_data)-1, :])
        data_array = np.append(data_array, new_data, axis = 0)
    
    
    print('\nSaving processed data...')
    data_array = data_array.reshape((len(dirlist), len(columns)))
    processed_data = pd.DataFrame(data_array, columns = columns)
    processed_data.to_csv(f'{args.results_dir}/processed_data.csv')
    print('Done!')



def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type = str,
        default = "../../results",
        help = "str variable, directory containing block data. Default: '../../results'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
