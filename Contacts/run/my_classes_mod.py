#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.special import softmax
from jax.tree_util import tree_map
import os, glob, sys
from subprocess import call
from os import listdir
from os.path import isfile, isdir

import torch
import esm

seed = 0
np.random.seed(seed)



### -------------------------------------- OLD ALGORITHM ------------------------------------- ###
class Old_class:
    ### Initialization
    def __init__(
            self,
            model, 
            alphabet,
            repr_layers
    ):

        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layers = repr_layers



    ### Calculate contact map through esm model
    def calculate_contacts(self, sequence, threshold = 0.5, return_binary = False):
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[self.repr_layers], return_contacts=True)
        token_len = batch_lens[0]
        attention_contacts = results["contacts"][0][:token_len, :token_len]

        if return_binary:
            return attention_contacts > threshold, attention_contacts
        else:
            return attention_contacts



### -------------------------------------- BASIC ALGORITHM ------------------------------------- ###
class Basic_class:

    ### Initialization
    def __init__(
            self,
            device : int = 0,
            distance_threshold : float = 4.
    ):

        torch.cuda.empty_cache()
    
        self.device = device
        torch.cuda.set_device(device)

        self.model = esm.pretrained.esmfold_v1()
        self.model = self.model.eval().cuda()

        self.distance_threshold = distance_threshold

        print('Basic class, status:')
        print(f'model: esmfold_v1')
        print(f'device: {self.device}')
        print(f'distance threshold: {self.distance_threshold} [A]', '\n')


    ### Calculate contact map through esmfold model
    def calculate_contacts(self, sequence, method = 'explicit', return_plddt = False):
        with torch.no_grad():
            output = self.model.infer(sequence)
        output = tree_map(lambda x: x.cpu().numpy(), output)

        plddt = output['plddt'][0, :, 1]
        plddt_mask = output['atom37_atom_exists'][0].astype(bool)        
        self._check_Ca(plddt_mask, sequence)

        if method == 'explicit':
            positions = output['positions'][-1, 0]
            positions_mask = output['atom14_atom_exists'][0].astype(bool)
            distance_matrix = self._calculate_distance_matrix(positions, positions_mask)
            contact_map = (distance_matrix < self.distance_threshold).astype(int)

        elif method == 'implicit':
            bins = np.append(0, np.linspace(2.3125, 21.6875, 63))
            contact_map = softmax(output['distogram_logits'], -1)[0]
            contact_map = contact_map[..., bins < 8].sum(-1)
            
        torch.cuda.empty_cache()
        del output
        
        if return_plddt:
            return contact_map, plddt
        else:
            return contact_map



    ### Calculate distance matrix for given chain
    def _calculate_distance_matrix(self, positions, positions_mask):
        distance_matrix = np.zeros( (len(positions), len(positions)) )
        idxs = np.arange(0, len(positions))
        for row in idxs:
            for col in idxs[idxs > row]:
                residue_one = positions[row, positions_mask[row]]
                residue_two = positions[col, positions_mask[col]]
                distance_matrix[row, col] = self._calculate_residue_distance(residue_one, residue_two)
                distance_matrix[col, row] = distance_matrix[row, col]
        return distance_matrix


    ### Calculate residue distance (minimum distance between atoms for the given residues)
    def _calculate_residue_distance(self, residue_one, residue_two):
        distances = []
        for xyz_one in residue_one:
            for xyz_two in residue_two:
                diff2_xyz = (xyz_one - xyz_two)**2
                distance = np.sqrt(np.sum(diff2_xyz))
                distances.append(distance)
        return np.min(distances)


    ### Check for missing C-alphas in the chain
    def _check_Ca(self, plddt_mask, sequence):
        check = np.all(plddt_mask[:, 1])
        assert check, fr'Missing C-$\alpha$ for loaded sequence: {sequence}'


    ### Set modules
    def set_device(self, device):
        self.device = device
        torch.cuda.set_device(device)

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold


    ### Get modules
    def get_device(self): return self.device
    def get_distance_threshold(self): return self.distance_threshold







### -------------------------------------- MUTATION ALGORITHM ------------------------------------- ###
class Mutation_class(Basic_class):

    ### Initialization
    def __init__(
            self,
            wt_sequence : str,
            ref_sequence : str = '',
            starting_sequence: str = '',
            metr_mutations : int = 100,
            eq_mutations : int = 0,
            T : float = 1.,
            gamma : float = 0.,
            results_dir : str = 'results',
            restart_bool : bool = False,
            device : int = 0,
            distance_threshold : float = 4.
    ):

        super().__init__(
                device = device,
                distance_threshold = distance_threshold
        )
        
        # Sequences
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)

        if ref_sequence == '':
            self.ref_sequence = self.wt_sequence
            self.ref_contacts = self.wt_contacts.copy()
        else:
            if len(ref_sequence) == len(wt_sequence): 
                self.ref_sequence = ref_sequence
                self.ref_contacts = self.calculate_contacts(self.ref_sequence)
            else: 
                raise ValueError("Mutation_class.__init__(): starting sequence ref_sequence must have the same length of the wild-type sequence.")
        self.ref_array = np.array(list(self.ref_sequence))

        if starting_sequence == '':
            self.starting_sequence = self.ref_sequence
            self.starting_contacts = self.ref_contacts.copy()
        else:
            if len(ref_sequence) == len(wt_sequence): 
                self.starting_sequence = starting_sequence
                self.starting_contacts = self.calculate_contacts(self.starting_sequence)
            else: 
                raise ValueError("Mutation_class.__init__(): starting sequence starting_sequence must have the same length of the wild-type sequence.")
        
        # Distance definitions
        self.distmatrix = pd.read_csv('inputs/DistPAM1.csv')
        self.distmatrix = self.distmatrix.drop(columns = ['Unnamed: 0'])
        self.residues = tuple(self.distmatrix.columns)
        self.distmatrix = np.array(self.distmatrix)

        # Parameters
        if metr_mutations > 0: self.metr_mutations = metr_mutations
        else: raise ValueError("Mutation_class.__init__(): metr_mutations must be positive.")

        if eq_mutations >= 0: self.eq_mutations = eq_mutations
        else: raise ValueError("Mutation_class.__init__(): eq_mutations can't be negative.")

        if T >= 0.: self.T = T
        else: raise ValueError("Mutation_class.__init__(): T can't be negative.")

        if gamma >= 0.: self.gamma = gamma
        else: raise ValueError("Mutation_class.__init__(): gamma can't be negative.")

        # Initialization
        self._get_id()
        self._check_directory(results_dir)
        self.set_restart_bool(restart_bool)
        self.print_status()



    ### Prepare simulation id
    def _get_id(self):
        T_str = str(self.T)
        g_str = str(self.gamma)
        self.file_id = 'T' + T_str + '_g' + g_str



    ### Check for directory to store simulation mutants and data
    def _check_directory(self, results_dir):
        if results_dir[-1] != '/' and results_dir[:3] != './' and results_dir[:4] != '../':
            self.results_dir = results_dir
        else:
            if results_dir[-1] == '/':
                self.results_dir = results_dir[:-1]
            if results_dir[:3] == './':
                self.results_dir = results_dir[3:]
            if results_dir[:4] == '../':
                self.results_dir = results_dir[4:]
        
        path = self.results_dir.split('/')
        actual_dir = '..'
        for idx, new_dir in enumerate(path):
            if idx > 0:
                actual_dir = actual_dir + '/' + path[idx - 1]
            onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
            if (new_dir in onlydirs) == False: 
                os.mkdir(f'{actual_dir}/{new_dir}')



    ### Reset parameters for new simulation
    def _reset(self):
        self.last_sequence = self.starting_sequence
        self.last_contacts = self.starting_contacts
        self.last_eff_energy = self.calculate_effective_energy(self.starting_contacts)
        self.last_ddG = 0
        self.last_PAM1_distance, self.last_Hamm_distance = self.get_distances(self.starting_sequence)
        self.generation = 0
        self.accepted_mutations = 0
        
        paths = [f'../{self.results_dir}/mutants_{self.file_id}.dat', f'../{self.results_dir}/data_{self.file_id}.dat', f'../{self.results_dir}/status_{self.file_id}.txt']
        onlyfiles = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]

        for path in paths:
            if path in onlyfiles:
                call(['rm', path])



    ### Restart the previous simulation
    def _restart(self):
        # Find files
        paths = [f'../{self.results_dir}/mutants_{self.file_id}.dat', f'../{self.results_dir}/data_{self.file_id}.dat']
        onlyfiles = [f'../{self.results_dir}/{f}' for f in listdir(f'../{self.results_dir}') if isfile(f'../{self.results_dir}/{f}')]
        check = np.all( [path in onlyfiles for path in paths] )

        if check:
            # Discard incomplete data
            with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'r') as mutants_file:
                muts_lines = mutants_file.readlines()
                muts_num = len(muts_lines)

            with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'r') as data_file:
                data_lines = data_file.readlines()
                data_num = len(data_lines)

            if muts_num < data_num:
                with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'w') as data_file:
                    for line in data_lines[:muts_num]: 
                        print(line, end = '', file = data_file)

            elif muts_num > data_num:
                with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'w') as mutants_file:
                    for line in muts_lines[:data_num]: 
                        print(line, end = '', file = mutants_file)

            # Last sequence residues and contacts
            with open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'r') as mutants_file:
                last_line = mutants_file.readlines()[-1].split('\t')
                self.last_sequence = last_line[1]
                if self.last_sequence[-1] == '\n': self.last_sequence = self.last_sequence[:-1]
                self.last_contacts = self.calculate_contacts(self.last_sequence)


            # Last sequence data
            with open(f'../{self.results_dir}/data_{self.file_id}.dat', 'r') as data_file:
                last_line = data_file.readlines()[-1].split('\t')
                self.generation = int( last_line[0] )
                self.last_eff_energy = float( last_line[1] )
                self.last_ddG = float( last_line[2] )
                self.last_PAM1_distance = float( last_line[3] )
                self.last_Hamm_distance = int( last_line[4] )
                self.accepted_mutations = int( float(last_line[5]) * self.generation )
                self.T = float( last_line[6] )
                self.gamma = float( last_line[7] )

        else:
            self._reset()



    ### Calculate Hamming distance and PAM1 distance from the reference sequence (ref_sequence)
    def get_distances(self, mt_sequence):
        mt_array = np.array(list(mt_sequence))
        new_residues_idxs = np.where(self.ref_array != mt_array)[0]

        # Hamming distance
        Hamm_distance = len(new_residues_idxs)

        # PAM1 distance
        old_residues = self.ref_array[new_residues_idxs]
        new_residues = mt_array[new_residues_idxs]
        PAM1_distance = 0.
        for old_residue, new_residue in zip(old_residues, new_residues):
            old_idx = self.residues.index(old_residue)
            new_idx = self.residues.index(new_residue)
            PAM1_distance += self.distmatrix[new_idx, old_idx]

        return PAM1_distance, Hamm_distance



    ### Produce single-residue mutation of the last metropolis sequence
    def single_mutation(self):
        # New residue
        position = np.random.randint(0, len(self.ref_sequence))
        residue = self.residues[ np.random.randint(0, len(self.residues)) ]

        if residue == self.last_sequence[position]:
            # Repeat if no mutation occurred
            mt_sequence = self.single_mutation()
            return mt_sequence

        else:
            # Generate mutant from last sequence
            mt_sequence = self.last_sequence[:position] + residue + self.last_sequence[(position + 1):]
            return mt_sequence



    ### Calculate effective as number of modified contacts divided by the number of the wild-type protein contacts
    def calculate_effective_energy(self, mt_contacts):
        # Modified contacts fraction
        mod_diff = abs(mt_contacts - self.wt_contacts)
        norm = np.sum(mt_contacts) + np.sum(self.wt_contacts)
        eff_en = np.sum(mod_diff) / norm
        return eff_en



    ### Calculate ddG
    def calculate_ddG(self):
        pass



    ### Metropolis algorithm
    def metropolis(self, equilibration = False, print_start = True):
        # Preparing the simulation
        if equilibration:
            mutations = self.eq_mutations
            mutants_file = open(f'../{self.results_dir}/eq_mutants_{self.file_id}.dat', 'w')
            data_file = open(f'../{self.results_dir}/eq_data_{self.file_id}.dat', 'w')
        else:
            mutations = self.metr_mutations
            mutants_file = open(f'../{self.results_dir}/mutants_{self.file_id}.dat', 'a')
            data_file = open(f'../{self.results_dir}/data_{self.file_id}.dat', 'a')

        if print_start:
            print(f'{self.generation}\t{self.last_sequence}', file = mutants_file)
            if self.generation == 0:
                print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{self.last_PAM1_distance}\t{self.last_Hamm_distance}\t{self.accepted_mutations}\t{self.T}\t{self.gamma}\t{len(self.wt_sequence)}', file = data_file)
            else:
                print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{self.last_PAM1_distance}\t{self.last_Hamm_distance}\t{self.accepted_mutations / self.generation}\t{self.T}\t{self.gamma}\t{len(self.wt_sequence)}', file = data_file)

        # Metropolis
        for imut in range(mutations):
            # Mutant generation
            self.generation += 1
            mt_sequence = self.single_mutation()
            mt_contacts = self.calculate_contacts(mt_sequence)
    
            # Observables
            eff_energy = self.calculate_effective_energy(mt_contacts)
            ddG = 0
            PAM1_distance, Hamm_distance = self.get_distances(mt_sequence)

            # Update lists
            if self.T == 0.:
                if eff_energy == 0.:
                    self.last_sequence = mt_sequence
                    self.last_contacts = mt_contacts
                    self.last_eff_energy = eff_energy
                    self.last_ddG = ddG
                    self.last_PAM1_distance = PAM1_distance
                    self.last_Hamm_distance = Hamm_distance
                    self.accepted_mutations += 1
            elif self.T > 0.:
                p = np.random.rand()
                if p < np.exp( - (eff_energy - self.last_eff_energy) / self.T  - self.gamma * (PAM1_distance - self.last_PAM1_distance)):
                    self.last_sequence = mt_sequence
                    self.last_contacts = mt_contacts
                    self.last_eff_energy = eff_energy
                    self.last_ddG = ddG
                    self.last_PAM1_distance = PAM1_distance
                    self.last_Hamm_distance = Hamm_distance
                    self.accepted_mutations += 1

            # Save data
            print(f'{self.generation}\t{self.last_sequence}', file = mutants_file)
            print(f'{self.generation}\t{format(self.last_eff_energy, ".15f")}\t{format(self.last_ddG, ".15f")}\t{self.last_PAM1_distance}\t{self.last_Hamm_distance}\t{self.accepted_mutations / self.generation}\t{self.T}\t{self.gamma}\t{len(self.ref_sequence)}', file = data_file)

            # Print and save last mutant
            if self.generation % 100 == 0:
                self.print_last_mutation(f'../{self.results_dir}/status_{self.file_id}.txt')

        # Close data files
        mutants_file.close()
        data_file.close()



    ### Print status
    def print_status(self):
        print(f'Simulation PID: {os.getpid()}\n')
        
        print(f'Mutation algorithm protein:')
        print(f'Wild-type sequence: {self.wt_sequence}')
        if self.ref_sequence != self.wt_sequence: 
            print(f'Reference sequence: {self.ref_sequence}\n')
        else: 
            print(f'Reference sequence: wild-type sequence\n')

        print(f'Mutation algorithm parameters:')
        print(f'metropolis mutations:    {self.metr_mutations}')
        print(f'equilibration mutations: {self.eq_mutations}')
        print(f'temperature:             {self.T}')
        print(f'gamma:                   {self.gamma}')
        print(f'distance threshold:      {self.distance_threshold} [A]')
        print(f'results directory:       ../{self.results_dir}\n')



    ### Print last mutation
    def print_last_mutation(self, print_file = sys.stdout):
        if print_file != sys.stdout:
            print_file = open(print_file, 'a')

        print(f'Generation:  {self.generation}', file = print_file)
        print(f'Wild tipe:   {self.wt_sequence}', file = print_file)
        if self.ref_sequence != self.wt_sequence:
            print(f'Reference sequence: {self.ref_sequence}', file = print_file)
        else:
            print(f'Reference sequence: wild-type sequence', file = print_file)
        
        print(f'Last mutant: {self.last_sequence}', file = print_file)
        print(f'Effective energy: {self.last_eff_energy}', file = print_file)
        print(f'ddG:              {self.last_ddG}', file = print_file)
        print(f'PAM1 distance:    {self.last_PAM1_distance}', file = print_file)
        print(f'Hamming distance: {self.last_Hamm_distance}\n', file = print_file)

        if print_file != sys.stdout:
            print_file.close()



    ### Set modules
    def set_wt_sequence(self, wt_sequence):
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        if len(self.ref_sequence) != len(self.wt_sequence):
            self.ref_sequence = self.wt_sequence
            self.ref_contacts = self.wt_contacts.copy()
        self._reset()

    def set_ref_sequence(self, ref_sequence):
        if len(ref_sequence) == len(self.wt_sequence):
            self.ref_sequence = ref_sequence
            self.ref_array = np.array(list(self.ref_sequence))
            self.ref_contacts = self.calculate_contacts(ref_sequence)
        else: 
            raise ValueError("Mutation_class.set_ref_sequence(): starting sequence ref_sequence must have the same length of the wild-type sequence.")

    def set_starting_sequence(self, starting_sequence):
        if len(starting_sequence) == len(self.wt_sequence):
            self.starting_sequence = starting_sequence
            self.starting_contacts = self.calculate_contacts(starting_sequence)
        else: 
            raise ValueError("Mutation_class.set_starting_sequence(): starting sequence starting_sequence must have the same length of the wild-type sequence.")

    def set_metr_mutations(self, metr_mutations): 
        if metr_mutations > 0: self.metr_mutations = metr_mutations
        else: raise ValueError("Mutation_class.set_metr_mutations(): metr_mutations must be positive.")

    def set_eq_mutations(self, eq_mutations):
        if eq_mutations >= 0: self.eq_mutations = eq_mutations
        else: raise ValueError("Mutation_class.set_eq_mutations(): eq_mutations can't be negative.")

    def set_T(self, T):
        if T >= 0.: self.T = T
        else: raise ValueError("Mutation_class.set_T(): T can't be negative.")
        self._get_id()

    def set_gamma(self, gamma):
        if gamma >= 0.: self.gamma = gamma
        else: raise ValueError("Mutation_class.set_gamma(): gamma can't be negative.")
        self._get_id()

    def set_restart_bool(self, restart_bool):
        self.restart_bool = restart_bool
        if self.restart_bool: self._restart()
        else: self._reset()



    ### Get modules
    def get_wt_sequence(self): return self.wt_sequence
    def get_wt_contacts(self): return self.wt_contacts
    def get_ref_sequence(self): return self.ref_sequence
    def get_ref_contacts(self): return self.ref_contacts
    def get_metr_mutations(self): return self.metr_mutations
    def get_eq_mutations(self): return self.eq_mutations
    def get_T(self): return self.T
    def get_gamma(self): return self.gamma
    def get_restart_bool(self): return self.restart_bool
    def get_generation(self): return self.generation
    def get_last_eff_energy(self): return self.last_eff_energy
    def get_last_ddG(self): return self.last_ddG
    def get_last_PAM1_distance(self): return self.last_PAM1_distance
    def get_last_Hamm_distance(self): return self.last_Hamm_distance
    def get_last_sequence(self): return self.last_sequence
    def get_last_contacts(self): return self.last_contacts
    def get_distmatrix(self): return self.distmatrix
    def get_residues(self): return self.residues
