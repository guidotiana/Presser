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

from utils import randomize_sequence


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
        print(f'model:              esmfold_v1')
        print(f'device:             {self.device}')
        print(f'distance threshold: {self.distance_threshold} [A]', '\n')


    ### Calculate contact map through esmfold model
    def calculate_contacts(self, sequence, method = 'explicit', return_plddt = False, return_trivial = False, return_distance = False):
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

        if not return_trivial:
            contact_map = self._eliminate_trivial_contacts(contact_map)

        output = {'contact_map': contact_map}
        if return_plddt: output['plddt'] = plddt
        if return_distance and method == 'explicit': output['distance_matrix'] = distance_matrix
        return output


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


    ### Eliminate trivial contacts from the contact map
    def _eliminate_trivial_contacts(self, contact_map):
        for row in range(len(contact_map)):
            contact_map[row, row] = 0
            if row > 0: contact_map[row, row - 1] = 0
            if row < len(contact_map) - 1: contact_map[row, row + 1] = 0
        return contact_map


    ### Set modules
    def set_device(self, device):
        self.device = device
        torch.cuda.set_device(device)

    def set_distance_threshold(self, distance_threshold):
        self.distance_threshold






### -------------------------------------- MUTATION ALGORITHM ------------------------------------- ###
class Mutation_class(Basic_class):

    ### Initialization
    def __init__(
            self,
            wt_sequence : str,
            ref_sequence : str = '',
            seed : int = 0,
            unique_length : int = 10000,
            results_dir : str = 'results',
            step : int = 1,
            restart : bool = False,
            device : int = 0,
            distance_threshold : float = 4.
    ):

        super().__init__(
                device = device,
                distance_threshold = distance_threshold
        )
        
        # Sequences
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)['contact_map']
        self.sequence_length = len(self.wt_sequence)
        
        if ref_sequence == '': self.ref_sequence = self.wt_sequence
        else:
            if len(ref_sequence) == self.length: self.ref_sequence = ref_sequence
            else: raise ValueError("Mutation_class.__init__(): ref_sequence must have the same length of the wt_sequence.")

        # Distance definitions
        self.PAM1_distance_matrix = pd.read_csv('inputs/DistPAM1.csv')
        self.PAM1_distance_matrix = self.PAM1_distance_matrix.drop(columns = ['Unnamed: 0'])
        self.residues = tuple(self.PAM1_distance_matrix.columns)
        self.PAM1_distance_matrix = np.array(self.PAM1_distance_matrix)

        # Parameters
        self.seed = seed
        np.random.seed(self.seed)

        if unique_length >= 0: self.unique_length = unique_length
        else: raise ValueError("Mutation_class.__init__(): unique_length can't be negative.")

        if step >= 1: self.step = step
        else: raise ValueError("Mutation_class.__init__(): step can't be less than 1.")

        self.restart = restart
        self.results_dir = results_dir


    ### Define starting sequence and associated observables
    def _starting_sequence(self):
        self.last_sequence = randomize_sequence(self.residues, self.wt_sequence, fraction = 1.)
        contacts = self.calculate_contacts(self.last_sequence)['contact_map']
        self.last_energy = self.calculate_energy(contacts)
        PAM1_distance, Hamm_distance = self.calculate_distances(self.last_sequence, self.ref_sequence)
        self.last_distance = {
                'PAM1': PAM1_distance,
                'Hamm': Hamm_distance
        }


    ### Reset parameters for new simulation
    def _reset(self, Tgt_file):
        self._starting_sequence()
        self.generation = 0
        self.am = 0
        
        self.unique_sequences = np.array([self.last_sequence], dtype = str)
        self.unique_energies = np.array([self.last_energy], dtype = float)

        saved_files = [f'{self.results_dir}/{f}' for f in listdir(f'{self.results_dir}') if isfile(f'{self.results_dir}/{f}')]
        for saved_file in saved_files:
            command = f'rm {saved_file}'.split(' ')
            call(command)

        command = f"cp {Tgt_file} {self.results_dir}".split(' ')
        call(command)


    ### Restart the previous simulation
    def _restart(self, Tgt_file):
        # Find files
        saved_files = [f for f in listdir(f'{self.results_dir}') if isfile(f'{self.results_dir}/{f}') and ('.dat' in f)]
        
        if len(saved_files) > 0:
            expected_files = ['data.dat', 'parameters.dat']
            for file in expected_files: assert file in saved_files, f'Expected missing file {file}.'
            assert len(expected_files) == len(saved_files), 'Wrong number of saved files.'

            # Discard incomplete data
            lines = {}
            lengths = []
            for f in saved_files:
                with open(f'{self.results_dir}/{f}', 'r') as ff:
                    lines[f] = ff.readlines()
                    lengths.append( len(lines[f]) )

            min_length = np.min(lengths)
            for f in saved_files:
                if len(lines[f]) > min_length:
                    with open(f'{self.results_dir}/{f}', 'w') as ff:
                        for line in lines[f][:min_length]:
                            print(line, end = '', file = ff)

            # Last sequence
            last_line = lines['data.dat'][-1]
            splitted_line = last_line.split('\t')
                
            self.last_sequence = splitted_line[1]
            self.last_energy = float(splitted_line[2])
            self.last_distance = {
                    'PAM1': float(splitted_line[3]),
                    'Hamm': float(splitted_line[4])
            }
            self.am = int(float(splitted_line[5]) * int(splitted_line[0]))

            # Unique replicas
            sequences = np.array([line.split('\t')[1] for line in lines['data.dat']])
            energies = np.array([float(line.split('\t')[2]) for line in lines['data.dat']])

            self.unique_sequences = np.unique(sequences)
            if len(self.unique_sequences) > self.unique_length:
                self.unique_sequences = self.unique_sequences[(len(self.unique_sequences) - self.unique_length):]

            idxs = [np.where(unique_sequence == sequences)[0][0] for unique_sequence in self.unique_sequences]
            self.unique_energies = np.array([energies[idx] for idx in idxs])
            assert len(self.unique_sequences) == len(self.unique_energies), 'Length mismatch between the unique lists.'

            # Parameters
            with open(f'{self.results_dir}/parameters.dat', 'r') as f:
                lines = f.readlines()
            last_line = lines[-1]
            splitted_line = last_line.split('\t')

            self.generation = int(splitted_line[0])
            self.T = float(splitted_line[1])
            self.gamma = float(splitted_line[2])
            self.threshold = float(splitted_line[3])
            self.seed = int(splitted_line[4])
            assert self.sequence_length == int(splitted_line[5]), 'Mismatch between saved and actual sequence length.'

            # Tgt_file
            with open(f'{self.results_dir}/Tgt_file.txt', 'r') as f:
                exp_lines = np.array( f.readlines() )
            with open(f'{Tgt_file}', 'r') as f:
                inp_lines = np.array( f.readlines() )
            assert np.all(exp_lines == inp_lines), 'Mutatioon_class._restart(): different saved and input Tgt_file.' 

        else:
            self._reset(Tgt_file)


    ### Calculate Hamming distance and PAM1 distance between sequences
    def calculate_distances(self, sequence_a, sequence_b):
        array_a = np.array(list(sequence_a))
        array_b = np.array(list(sequence_b))
        new_residues_idxs = np.where(array_a != array_b)[0]

        # Hamming distance
        Hamm_distance = len(new_residues_idxs) / self.sequence_length

        # PAM1 distance
        residues_a = array_a[new_residues_idxs]
        residues_b = array_b[new_residues_idxs]
        PAM1_distance = 0.
        for residue_a, residue_b in zip(residues_a, residues_b):
            idx_a = self.residues.index(residue_a)
            idx_b = self.residues.index(residue_b)
            PAM1_distance += self.PAM1_distance_matrix[idx_a, idx_b]
        PAM1_distance = PAM1_distance / self.sequence_length
        
        return PAM1_distance, Hamm_distance


    ### Produce single-residue mutation of the last metropolis sequence
    def single_mutation(self, sequence):
        # New residue
        position = np.random.randint(0, self.sequence_length)
        residue = self.residues[ np.random.randint(0, len(self.residues)) ]

        if residue == sequence[position]:
            # Repeat if no mutation occurred
            mt_sequence = self.single_mutation(sequence)
            return mt_sequence

        else:
            # Generate mutant from last sequence
            mt_sequence = sequence[:position] + residue + sequence[(position + 1):]
            return mt_sequence


    ### Calculate effective as number of modified contacts divided by the number of the wild-type protein contacts
    def calculate_energy(self, mt_contacts):
        # Modified contacts fraction
        mod_diff = abs(mt_contacts - self.wt_contacts)
        norm = np.sum(mt_contacts) + np.sum(self.wt_contacts)
        energy = np.sum(mod_diff) / norm
        return energy


    ### Calculate ddG
    def calculate_ddG(self):
        pass


    ### Metropolis algorithm
    def _metropolis(self, save, print_progress):
        # Starting status print
        if save and self.generation == 0: self._save()
        if print_progress: self.print_progress()

        # Metropolis
        while self.last_energy > self.threshold:
            # Mutant generation
            self.generation += 1
            mt_sequence = self.single_mutation(self.last_sequence)

            # Observables
            mask = self.unique_sequences == mt_sequence
            assert np.sum(mask.astype(int)) <= 1, "Too many 'unique' sequences equal to the same mutant."
            if np.any(mask):
                assert self.unique_sequences[mask][0] == mt_sequence, 'Wrong mask.'
                mt_energy = self.unique_energies[mask][0]
                self.visited_counter += 1
            else:
                mt_contacts = self.calculate_contacts(mt_sequence)['contact_map']
                mt_energy = self.calculate_energy(mt_contacts)

                self.unique_sequences = np.append(self.unique_sequences, mt_sequence)
                self.unique_energies = np.append(self.unique_energies, [mt_energy])
                assert len(self.unique_sequences) == len(self.unique_energies), "Length of unique sequences and unique energies must coincide."
                if len(self.unique_sequences) > self.unique_length:
                    self.unique_sequences = self.unique_sequences[1:]
                    self.unique_energies = self.unique_energies[1:]
            
            PAM1_distance, Hamm_distance = self.calculate_distances(mt_sequence, self.ref_sequence)
            new_distance = {
                    'PAM1': PAM1_distance,
                    'Hamm': Hamm_distance
            }

            # Update lists
            dE = mt_energy - self.last_energy
            dd = new_distance['PAM1'] - self.last_distance['PAM1']
            p = np.random.rand()
            if p <= np.exp( -dE/self.T -self.gamma*dd ):
                self.last_sequence = mt_sequence
                self.last_energy = mt_energy
                for key in self.last_distance: self.last_distance[key] = new_distance[key]
                self.am += 1

            # Save data
            if save and (self.generation%self.step == 0): self._save()
            if print_progress and (self.generation%1000 == 0): self.print_progress()


    ### Start simulation based on Tgm_file
    def metropolis(self, Tgt_file, save = True, print_progress = True):
        if self.restart: self._restart(Tgt_file)
        else: self._reset(Tgt_file)

        ## Simulation parameters
        with open(Tgt_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 1, f'Too many lines for quench algorithm. Expected lines 1, Found lines {len(lines)}.'
        T, gamma, threshold = float(lines[0].split('\t')[0]), float(lines[0].split('\t')[1]), float(lines[0].split('\t')[2])
        self.T = T
        self.gamma = gamma
        self.threshold = threshold

        self.visited_counter = 0
        self.print_status()
        self._metropolis(save, print_progress)


    ### Print status
    def print_status(self):
        print(f'Mutation algorithm protein:')
        print(f'Wild-type sequence: {self.wt_sequence}')
        print(f'Reference sequence: {self.ref_sequence}')
        print(f'Starting sequence:')
        if self.generation == 0: print(f'{self.last_sequence}\t{self.last_energy}\t{self.last_distance["Hamm"]}\t{self.am}')
        else: print(f'{self.last_sequence}\t{self.last_energy}\t{self.last_distance["Hamm"]}\t{self.am/self.generation}')
        print()

        print(f'Mutation algorithm parameters:')
        print(f'current generation:  {self.generation}')
        print(f'temperature:         {self.T}')
        print(f'gamma:               {self.gamma}')
        print(f'threshold:           {self.threshold}')
        print(f'seed:                {self.seed}')
        print(f'unique length:       {self.unique_length}')
        print(f'results directory:   {self.results_dir}')
        print(f'step:                {self.step}')
        print(f'restart:             {self.restart}\n')


    ### Print simulation progress
    def print_progress(self):
        print(f'- Generation: {self.generation}')
        print(f'- Visited counter: {self.visited_counter}')
        print(f'- Last sequence:')
        if self.generation == 0: print(f'{self.last_sequence}\t{self.last_energy}\t{self.last_distance["Hamm"]}\t{self.am}')
        else: print(f'{self.last_sequence}\t{self.last_energy}\t{self.last_distance["Hamm"]}\t{self.am/self.generation}')
        print()


    ### Save data
    def _save(self):
        # Save data
        with open(f'{self.results_dir}/data.dat', 'a') as f:
            line = f'{self.generation}\t'
            line = line + f'{self.last_sequence}\t'
            line = line + f'{self.last_energy}\t'
            line = line + f'{self.last_distance["PAM1"]}\t'
            line = line + f'{self.last_distance["Hamm"]}\t'
            if self.generation == 0: line = line + f'{self.am}'
            else: line = line + f'{self.am/self.generation}'
            print(line, file = f)

        # Save parameters
        with open(f'{self.results_dir}/parameters.dat', 'a') as f:
            line = f'{self.generation}\t'
            line = line + f'{self.T}\t'
            line = line + f'{self.gamma}\t'
            line = line + f'{self.threshold}\t'
            line = line + f'{self.seed}\t'
            line = line + f'{self.sequence_length}'
            print(line, file = f)


    ### Set modules
    def set_wt_sequence(self, wt_sequence : str):
        self.wt_sequence = wt_sequence
        self.wt_contacts = self.calculate_contacts(self.wt_sequence)
        self.sequence_length = len(self.wt_sequence)

    def set_ref_sequence(self, ref_sequence : str):
        if len(ref_sequence) == self.sequence_length: self.ref_sequence = ref_sequence
        else: raise ValueError("Mutation_class.set_ref_sequence(): ref_sequence must have the same length of the wt_sequence.")

    def set_seed(self, seed : int):
        self.seed = seed
        np.random.seed(self.seed)

    def set_unique_length(self, unique_length : int):
        if unique_length >= 0: self.unique_length = unique_length
        else: raise ValueError("Mutation_class.set_unique_length(): unique_length can't be negative.")

    def set_results_dir(self, results_dir : str):
        self.results_dir = results_dir

    def set_step(self, step : int):
        if step >= 1: self.step = step
        else: raise ValueError("Mutation_class.set_step(): step can't be less than 1.")

    def set_restart(self, restart : bool):
        self.restart = restart
