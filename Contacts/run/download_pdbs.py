#!/usr/bin/env python3
import numpy as np
from tqdm import tqdm
import os
from os import listdir
from os.path import isdir, isfile, join
import Bio.PDB as PDB
import subprocess
import argparse


def main(args):
	# Paths and stderr file
	lists_path = '../lists'
	files_path = '../files'
	
	# Load initial pdb list
	print('Cleaning downloaded raw pdb list...')
	lines = open(f'{lists_path}/raw_pdb_list.txt', 'r').readlines()
	raw_pdb_codes = []
	for line in lines:
		if line[-1] == '\n': raw_pdb_codes.append( line[:-2] )
		else: raw_pdb_codes.append( line[:-1] )
	raw_pdb_codes = np.array(raw_pdb_codes)
	
	# Drop more-than-one-chain pdbs in the raw list
	unique_pdb_codes, counts = np.unique(raw_pdb_codes, return_counts = True)
	mask = counts > 1
	multi_chain_pdb_codes = unique_pdb_codes[mask]
	pdb_codes = [code for code in raw_pdb_codes if not code in multi_chain_pdb_codes]
	
	# Downloading pdb files
	if args.download:
		print('Downloading pdb files...')
		for pdb_code in tqdm(pdb_codes, total = len(pdb_codes)):
			subprocess.call(['wget', f'https://files.rcsb.org/view/{pdb_code}.pdb'])
			subprocess.call(['mv', f'{pdb_code}.pdb', files_path])
			
	# Check for directories in 'files_path'
	print('Checking for directories...')
	bad_dir = 'bad_files'
	multi_chain_dir = 'multi_chain_files'
	onlydirs = [d for d in listdir(files_path) if isdir(files_path + '/' + d)]
	if not bad_dir in onlydirs: 
		os.mkdir(files_path + '/' + bad_dir)
	if not multi_chain_dir in onlydirs: 
		os.mkdir(files_path + '/' + multi_chain_dir)
	
	# Complete cleaning
	print('Eliminating bad pdb files, more-than-one-chain proteins and unknown residue chains...')
	final_pdb_codes = []
	bad_counter, multi_chain_counter = 0, 0
	for pdb_code in tqdm(pdb_codes, total = len(pdb_codes)):
		list_of_commands = f'./warnings/simple_code.py --pdb-code {pdb_code}'.split(' ')
		subprocess.call(list_of_commands)
			
		# Check for warnings
		file = open('warnings/stderr.out', 'r')
		lines = file.readlines()
		file.close()
		
		ConstructionWarning = np.any(['PDBConstructionWarning' in line for line in lines])
		ChainWarning = np.any(['ChainWarning' in line for line in lines])
		ResidueWarning = np.any(['ResidueWarning' in line for line in lines])
		
		if ConstructionWarning or ResidueWarning:
			subprocess.call(["mv", f"{files_path}/{pdb_code}.pdb", f'{files_path}/{bad_dir}'])
			bad_counter += 1
		elif ChainWarning:
			subprocess.call(["mv", f"{files_path}/{pdb_code}.pdb", f'{files_path}/{multi_chain_dir}'])
			multi_chain_counter += 1
		else:
			final_pdb_codes.append(pdb_code)
	
	# Save cleaned pdbs
	file = open(f'{lists_path}/pdb_list.txt', 'w')
	for code in final_pdb_codes: print(f'{code}\n', file = file, end = '')
	file.close()
	
	# Print cleaning process status
	print()
	print(f'Final PDB codes: {len(final_pdb_codes)}')
	print(f'Bad PDB codes (ConstructionWarning or ResidueWarning): {bad_counter}')
	print(f'Multi-chain PDB codes (ChainWarning): {multi_chain_counter}', end = '\n\n')
	

def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--download",
		type = bool,
		default = False,
		help = "boolean variable, to download or not the pdb files. Default: False'."
	)
	return parser
    
if __name__ == '__main__':
	parser = create_parser()
	args = parser.parse_args()
	main(args)
