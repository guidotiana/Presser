#!/usr/bin/env python
import numpy as np
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
import Bio.PDB as PDB
import argparse
import time

import torch
import esm

from my_classes import *


names = ['ALA', 'ASX', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'XAA', 'TYR', 'GLX']
symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
	
def modify_list(pdb_list, results_path):
    """Eliminate already studied proteins from the pdb list"""
    filenames = [filename for filename in listdir(results_path) if isfile(join(results_path, filename))]
    done_pdbs = []
    
    for filename in filenames:
        with open(f'{results_path}/{filename}', 'r') as file:
            lines = file.readlines()
        if len(lines) > 0:
            if filename == 'data.txt': splitted_lines = [str(line.split('\t')[0])[:-2] for line in lines]
            elif filename == 'skipped.txt': splitted_lines = [str(line.split('\t')[0]) for line in lines]
            done_pdbs = done_pdbs + splitted_lines

    for done_pdb in done_pdbs:
        if done_pdb in pdb_list:
            pdb_list.remove(done_pdb)
    return pdb_list

def check_model(model, args):
    """Check for number of chains, chain length and residue names"""
    model_check = len(model) == 1

    chain = model.child_list[0]
    chain_name = list(model.child_dict.keys())[model.child_list.index(chain)]
    modified_chain = drop_het_residues(chain)

    residue_names = [residue.resname for residue in modified_chain]
    length_check = len(residue_names) < args.max_residues
    names_check = np.all([residue.resname in names for residue in modified_chain])

    if model_check and length_check and names_check:        # do not skip
        return modified_chain, chain_name, residue_names
    else:                                                   # skip
        return None

def drop_het_residues(chain):
    """Returns a chain without the het residues"""
    dropped_residue_ids = [residue.get_id() for residue in chain if residue.get_id()[0] != ' ']
    for dropped_residue_id in dropped_residue_ids:
        chain.detach_child(dropped_residue_id)
    return chain
	
def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    coords_one = [residue_one[name].coord for name in residue_one.child_dict if not list(name)[0] == 'H']
    coords_two = [residue_two[name].coord for name in residue_two.child_dict if not list(name)[0] == 'H']
    
    distances = []
    for coord_one in coords_one:
        for coord_two in coords_two:
            diff_vector  = coord_one - coord_two
            distance = np.sqrt(np.sum(diff_vector * diff_vector))
            distances.append(distance)
    return np.min(distances)
	
def calc_dist_matrix(chain) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain), len(chain)), float)
    for row, residue_one in enumerate(chain) :
        for col, residue_two in enumerate(chain) :
            if col > row:
                answer[row, col] = calc_residue_dist(residue_one, residue_two)
                answer[col, row] = answer[row, col]
    return answer
    
def calc_eff_energy(PDB_contact_map, esmfold_contact_map):
    mod_diff = abs(PDB_contact_map - esmfold_contact_map)
    norm = np.sum(PDB_contact_map) + np.sum(esmfold_contact_map)
    eff_energy = np.sum(mod_diff) / norm
    return eff_energy
	
	

def main(args):
    print(f'pid: {os.getpid()}')

    # Collect pdbs
    print('Collecting pdbs...')
    files_path = '../files'
    results_path = '../results'
    
    pdb_list = [file[:-4] for file in listdir(files_path) if isfile(join(files_path, file)) and file[-3:] == 'pdb']
    print(f'  -total files:     {len(pdb_list)}')
    if args.restart: 
        pdb_list = modify_list(pdb_list, results_path)
    print(f'  -remaining files: {len(pdb_list)}', '\n')
    
    # Load model
    print('Loading esmfold model...')
    predictor = Basic_class(
            device = args.device,
            distance_threshold = args.distance_threshold
    )


    if args.restart: mode = 'a'
    else: mode = 'w'
    data_file = open(f'{results_path}/data.txt', mode)
    skipped_file = open(f'{results_path}/skipped.txt', mode)

    for pdb_code in tqdm(pdb_list, total = len(pdb_list)):
        pdb_filename = f'{files_path}/{pdb_code}.pdb'
        structure = PDB.PDBParser().get_structure(pdb_code, pdb_filename)
        model = structure[0]

        # Check and print protein properties
        output = check_model(model, args)
        if output == None:
            print(f'PDB {pdb_code}, skipped.')
            print(pdb_code, file = skipped_file)
            continue
        else:
            chain, chain_name, residue_names = output

        idxs = [names.index(name) for name in residue_names]
        wt_sequence_list = [symbols[idx] for idx in idxs]
        wt_sequence = ''.join(wt_sequence_list)
        print(f'PDB {pdb_code}, Chain {chain_name}, {len(wt_sequence)} residues.')

        # Calculate PDB contact map
        print('Calculating PDB contact map...')
        dist_matrix = calc_dist_matrix(chain)
        PDB_contact_map = (dist_matrix < args.distance_threshold).astype(int)
        contact_number = int( (np.sum(PDB_contact_map) - len(wt_sequence)) / 2 )

        # Calculate contact map through distance calculation
        print('Calculating esmfold predicted contact map, explicit method...')
        t0 = time.time()
        ex_esmfold_contact_map, plddt = predictor.calculate_contacts(wt_sequence, method = 'explicit', return_plddt = True)
        ex_dt = time.time() - t0
        ex_eff_energy = calc_eff_energy(PDB_contact_map, ex_esmfold_contact_map)

        if args.print:
            print(f'eff_energy: {format(ex_eff_energy, ".5f")}\t{format(ex_dt, ".2f")} s')
        
        # Calculate contact map through distogram_logits
        print('Calculating esmfold predicted contact map, implicit method...')
        t0 = time.time()
        im_esmfold_contact_map = predictor.calculate_contacts(wt_sequence, method = 'implicit')
        im_dt = time.time() - t0
        im_eff_energy = calc_eff_energy(PDB_contact_map, im_esmfold_contact_map)

        if args.print:
            print(f'eff_energy: {format(im_eff_energy, ".5f")}\t{format(im_dt, ".2f")} s')


        # Print data
        if args.print:
            print(f'Mean (Ca) plddt: {format(plddt.mean(), ".4f")}', '\n')


        # Save data
        print('Saving data...\n')
        print(f'{pdb_code}_{chain_name}\t{len(wt_sequence)}\t{contact_number}\t{args.distance_threshold}\t{ex_eff_energy}\t{ex_dt}\t{im_eff_energy}\t{im_dt}\t{plddt.mean()}\n', file = data_file, end = '')
        if args.save_cm:
            np.save(join(results_path, f'{pdb_code}_{chain_name}_PDB_cm.npy'), np.array(PDB_contact_map))
            np.save(join(results_path, f'{pdb_code}_{chain_name}_ex_esmfold_cm.npy'), np.array(ex_esmfold_contact_map))
            np.save(join(results_path, f'{pdb_code}_{chain_name}_im_esmfold_cm.npy'), np.array(im_esmfold_contact_map))

    data_file.close()
    skipped_file.close()


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type = int,
        default = 0,
        help = "int variable, cuda device used to load the esmfold model. Possible choices: 0, 1, 2, 3. Default: 0."
    )
    parser.add_argument(
        "--restart",
        type = bool,
        default = True,
        help = "boolean variable, restart (True) or don't restart (False) from the last analyzed sequence. Default: True."
    )
    parser.add_argument(
        "--max-residues",
        type = int,
        default = 500,
        help = "int variable, maximum protein length. Default: 500 [residues]."
    )
    parser.add_argument(
    	"--distance-threshold",
    	type = float,
    	default = 4.,
    	help = "float variable, maximum distance which defines a contact. Default: 4.0 [A]."
    )
    parser.add_argument(
        "--print",
        type = bool,
        default = True,
        help = "boolean variable, print (True) or don't print (False) the calculated eff_energy. Default: True."
    )
    parser.add_argument(
        "--save-cm",
        type = bool,
        default = False,
        help = "boolean variable, save (True) or don't save (False) the PDB contact map and the esmfold predicted contact map. Default: False."
    )
    return parser	


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)
