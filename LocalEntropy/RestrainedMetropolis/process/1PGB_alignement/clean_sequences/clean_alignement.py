import argparse
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, isdir


def main(args):
    # Reading input file
    with open(args.raw_file, 'r') as f:
        lines = f.readlines()
    
    # Discarding non-sequence data
    for line in lines:
        if line[0] == '#' or line[0] == '/': lines.pop(lines.index(line))

    # Collecting sequences
    sequences, codes = [], []
    for iline, line in enumerate(lines):
        codes.append(line.split(' ')[0])
        sequences.append(line.split(' ')[-1][:-1])
    
    # Uniforming lengths
    max_length = np.max([len(sequence) for sequence in sequences])
    for iseq in range(len(sequences)):
        while len(sequences[iseq]) < max_length:
            sequences[iseq] = sequences[iseq] + '-'
    assert len(np.unique([len(sequence) for sequence in sequences])) == 1, 'Non-matching lengths.'
    
    # Cleaning non-residue columns
    wt_seq = sequences[0]
    bad_idxs = [idx for idx in range(len(wt_seq)) if (wt_seq[idx] == '-') or (wt_seq[idx] == '.')]
    with open('1PGB_alignement.txt', 'w') as f:
        for iseq, (sequence, code) in enumerate(zip(sequences, codes)):
            sequences[iseq] = ''.join([sequence[idx] for idx in range(len(sequence)) if (not idx in bad_idxs)])
            print(f'{iseq}\t{code}\t{sequences[iseq]}', file = f)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-file",
        type = str,
        default = "raw_1PGB_alignement.uniprot",
        help = "str variable, file containing raw sequences to align. Default: 'raw_1PGB_alignement.uniprot'."
    )
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
