import argparse
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, isdir

from utils import load_inputs, load_data


def main(args):
    print('Loading parameters...')
    tuples = [
        ('clean', bool),
        ('raw_data_directory', str),
        ('cleaned_data_directory', str),
        ('ob_idx', int),
        ('ebin', float),
        ('emin', float),
        ('tbin', float),
        ('tmin', float),
        ('ntbin', int),
    ]
    parameters = load_inputs(args.input_file, tuples)

    if parameters["clean"]:
        print('Cleaning mutants...')
        filenames, Ts = [], []
        dirlist = [d for d in listdir(parameters["raw_data_directory"]) if isdir(f'{parameters["raw_data_directory"]}/{d}')]
        for d in dirlist:
            print(f'- {d} directory...')
            block_file = [f for f in listdir(f'{parameters["raw_data_directory"]}/{d}') if isfile(f'{parameters["raw_data_directory"]}/{d}/{f}') and ('block_data' in f)]
            if len(block_file) == 0:
                print('skipped, no block_data file found.')
            else:
                assert len(block_file) == 1, 'Too many block_data files.'
                block_file = block_file[0]
                block_data = pd.read_csv(f'{parameters["raw_data_directory"]}/{d}/{block_file}')

                (_, __, T_string, gamma_string) = d.split('_')
                raw_filename = f'data_{T_string}_{gamma_string}'
                seed_ds = [subd for subd in listdir(f'{parameters["raw_data_directory"]}/{d}') if isdir(f'{parameters["raw_data_directory"]}/{d}/{subd}')]
                discarded_mutations = int(block_data.loc[0, 'discarded_mutations'] / len(seed_ds))
                for iseed, seed_d in enumerate(seed_ds):
                    print(f'  ... {seed_d} seed directory ({iseed + 1}/{len(seed_ds)})')
                    actual_d = f'{parameters["raw_data_directory"]}/{d}/{seed_d}'
                    seed_data = load_data(actual_d, return_eq = False)
                    length = seed_data[0, -1]
                    seed_data = seed_data[discarded_mutations:, [0, 1, 3, 4]]
                    seed_data[:, [2, 3]] = seed_data[:, [2, 3]] / length
                
                    seed_filename = f'{raw_filename}_{seed_d}.dat'
                    with open(f'{parameters["cleaned_data_directory"]}/{seed_filename}', 'w') as f:
                        for line in seed_data:
                            print(f'{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}', file = f)

                    filenames.append(f'{parameters["cleaned_data_directory"]}/{seed_filename}')
                    Ts.append(float(T_string[1:]))
                print('  done!')
    else:
        with open('mhistogram.in', 'r') as f:
            lines = f.readlines()
        ntemp = int(lines[0].split(' ')[1])
        lines = lines[2 : ntemp + 2]
        splitted_lines = np.array([line.split(' ') for line in lines])
        filenames, Ts = splitted_lines[:, 0], splitted_lines[:, 1]


    print('Writing mhistogram.in ...')
    sorted_Ts = np.sort(Ts)[::-1]
    with open('mhistogram.in', 'w') as f:
        print(f'ntemp {len(filenames)}', file = f)
        print('files', file = f)
        for T in sorted_Ts:
            for filename in filenames:
                if str(T) in filename:
                    print(f'{filename} {T} {parameters["ob_idx"]}', file = f)
                    filenames.remove(filename)
        print('', file = f)
        assert len(filenames) == 0, 'Simulation file left out.'

        print('nohisto', file = f)
        for key in ('emin', 'ebin', 'tmin', 'tbin', 'ntbin'):
            print(f'{key} {parameters[key]} ', file = f)
    print('Done!')



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
