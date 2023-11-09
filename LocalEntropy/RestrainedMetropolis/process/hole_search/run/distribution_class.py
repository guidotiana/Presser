import numpy as np
import pandas as pd
from time import time
import os
from os import listdir
from os.path import isfile, isdir

from utils import *



class Distribution_class:
    ### Initialization
    def __init__(
            self,
            dtype : str = 'Hamm',
            weighted : bool = True,
            inputs_d : str = 'inputs',
            prec : float = 0.
    ):

        if not dtype in ("Hamm", "PAM1"): raise ValueError("Incompatible value for dtype variable. Allowed values: 'Hamm' (Hamming distance pdf), 'PAM1' (PAM1 distance pdf).")
        else:
            self.dtype = dtype
            if self.dtype == "PAM1":
                distmatrix = pd.read_csv(f'{self.inputs_d}/DistPAM1.csv')
                distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
                self.residues = tuple(distmatrix.columns)
                self.distmatrix = np.array(distmatrix)

        self.inputs_d = inputs_d
        self.weighted = weighted
        self.site_weights = np.load(f'{self.inputs_d}/site_weights.npy')
        if not weighted:
            self.site_weights = np.ones(len(self.site_weights))

        if prec < 0.: raise ValueError("Incompatible value for prec variable. Allowed values: prec > 0.")
        elif prec == 0.: self.prec = np.min(self.site_weights)
        else: self.prec = prec



    ### Check mutants for function calculation
    def _check_mutants(self, mutants):
        unique_lengths = np.unique([len(mutant) for mutant in mutants])
        assert len(unique_lengths) == 1, "Can't calculate distributions from sequences with different lengths."
        self.length = unique_lengths[0]
        
        unique_mutants, self.counts = np.unique(mutants, return_counts = True)
        self.mutants = np.array([list(mutant) for mutant in unique_mutants], dtype = str)



    ### Calculate PAM1 distance between sequences
    def calculate_weighted_PAM1_distance(self, mut1, mut2):
        diff_residues_idxs = np.where(mut1 != mut2)[0]
        
        weights = self.site_weights[diff_residues_idxs]
        diff_residues1 = mut1[diff_residues_idxs]
        diff_residues2 = mut2[diff_residues_idxs]
        PAM1_distance = 0.
        for residue1, residue2, weight in zip(diff_residues1, diff_residues2, weights):
            idx1 = self.residues.index(residue1)
            idx2 = self.residues.index(residue2)
            PAM1_distance += weight * self.distmatrix[idx1, idx2]

        return PAM1_distance



    ### Calculate weighted Hamm distance between sequences
    def calculate_weighted_Hamm_distance(self, mut1, mut2):
        diff_residues_idxs = np.where(mut1 != mut2)[0]
        weights = self.site_weights[diff_residues_idxs]
        Hamm_distance = weights.sum()
        return Hamm_distance



    ### Calculate pdf
    def _calculate_pdf(self, mutants):
        self._check_mutants(mutants)
        t0, partial_t0 = time(), time()

        max_distance = self.site_weights.sum()
        pdf = np.zeros(int(max_distance / self.prec))

        for imut_a, (mut_a, count_a) in enumerate(zip(self.mutants, self.counts)):
            if imut_a % 1000 == 0:                    
                print(f'progress: {imut_a}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
                partial_t0 = time()

            partial_counts = self.counts.copy()
            partial_counts[imut_a] -= 1
            partial_pdf = np.zeros(int(max_distance / self.prec))

            for (mut_b, count_b) in zip(self.mutants, partial_counts):
                if self.dtype == 'PAM1': distance_ab = self.calculate_weighted_PAM1_distance(mut_a, mut_b)
                else: distance_ab = self.calculate_weighted_Hamm_distance(mut_a, mut_b)
                idistance_ab = int( (distance_ab - distance_ab % self.prec) / self.prec )
                if idistance_ab == int(max_distance / self.prec): idistance_ab = idistance_ab - 1
                partial_pdf[idistance_ab] += count_b

            pdf = pdf + partial_pdf * count_a

        print(f'progress: {len(self.mutants)}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
        print(f'Total time: {format(time() - t0, ".1f")}')
        
        pdf = pdf / np.sum(pdf)
        distances = np.arange(len(pdf)) * self.prec
        q = abs(self.site_weights.sum() - distances)
        results = pd.DataFrame({
            'distances': distances,
            'q': q,
            'pdf': pdf
        })
        return results



    ### Calculate pdf, simple Hamming
    def _calculate_simple_pdf(self, mutants):
        self._check_mutants(mutants)
        t0, partial_t0 = time(), time()

        pdf = np.zeros(self.length + 1)
        for imut_a, (mut_a, count_a) in enumerate(zip(self.mutants, self.counts)):
            if imut_a % 1000 == 0:
                print(f'progress: {imut_a}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
                partial_t0 = time()

            partial_counts = self.counts.copy()
            partial_counts[imut_a] -= 1
            partial_pdf = np.zeros(self.length + 1)

            for (mut_b, count_b) in zip(self.mutants, partial_counts):
                distance_ab = len(np.where(mut_a != mut_b)[0])
                partial_pdf[distance_ab] += count_b

            pdf = pdf + partial_pdf * count_a

        print(f'progress: {len(self.mutants)}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
        print(f'Total time: {format(time() - t0, ".1f")}')

        pdf = pdf / np.sum(pdf)
        scaled_distances = np.arange(len(pdf)) * self.prec / self.site_weights.sum()
        q = abs(scaled_distances.max() - scaled_distances)
        results = pd.DataFrame({
            'distances': scaled_distances,
            'q': q,
            'pdf': pdf
        })
        return results



    ### Calculate characteristic dimension
    def _calculate_cd(self, mutants, ref_mutant, return_pdf = False):
        self._check_mutants(mutants)
        ref_mutant = np.array(list(ref_mutant))
        assert len(ref_mutant) == self.length, "Can't calculate distributions from sequences with different lengths."
        mask = [np.all(mutant == ref_mutant) for mutant in self.mutants]
        self.counts[mask] -= 1
        t0, partial_t0 = time(), time()

        max_distance = self.site_weights.sum()
        ref_pdf = np.zeros(int(max_distance / self.prec))
        for imut, (mutant, count) in enumerate(zip(self.mutants, self.counts)):
            if imut % 1000 == 0:                    
                print(f'progress: {imut}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
                partial_t0 = time()

            if self.dtype == 'PAM1': distance = self.calculate_weighted_PAM1_distance(mutant, ref_mutant)
            else: distance = self.calculate_weighted_Hamm_distance(mutant, ref_mutant)
            idistance = int( (distance - distance % self.prec) / self.prec )
            if idistance == int(max_distance / self.prec): idistance = idistance - 1
            ref_pdf[idistance] += count

        print(f'progress: {len(self.mutants)}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
        print(f'Total time: {format(time() - t0, ".1f")}')

        distances = np.arange(len(ref_pdf)) * self.prec
        masked_ref_pdf = ref_pdf[distances > 1.]
        masked_ref_cdf = np.array([masked_ref_pdf[:(idx + 1)].sum() for idx in range(len(masked_ref_pdf))])
        masked_ref_cdf[masked_ref_cdf == 0.] = 1.
        cd = np.log(masked_ref_cdf) / np.log(distances[distances > 1.])
        cd = np.append([0.] * len(distances[distances <= 1.]), cd)

        ref_mutant = ''.join(ref_mutant)
        q = abs(self.site_weights.sum() - distances)
        if return_pdf:
            ref_pdf = ref_pdf / np.sum(ref_pdf)
            results = pd.DataFrame({
                'distances': distances,
                'q': q,
                'cd': cd,
                'ref_pdf': ref_pdf,
                'ref_mutant': [ref_mutant] * len(distances)
            })
        else:
            results = pd.DataFrame({
                'distances': distances,
                'q': q,
                'cd': cd,
                'ref_mutant': [ref_mutant] * len(distances)
            })
        return results



    ### Calculate characteristic dimension, simple Hamming
    def _calculate_simple_cd(self, mutants, ref_mutant, return_pdf = False):
        self._check_mutants(mutants)
        ref_mutant = np.array(list(ref_mutant))
        assert len(ref_mutant) == self.length, "Can't calculate distributions from sequences with different lengths."
        mask = [np.all(mutant == ref_mutant) for mutant in self.mutants]
        self.counts[mask] -= 1
        t0, partial_t0 = time(), time()

        ref_pdf = np.zeros(self.length + 1)
        for imut, (mutant, count) in enumerate(zip(self.mutants, self.counts)):
            if imut % 1000 == 0:
                print(f'progress: {imut}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
                partial_t0 = time()

            distance = len(np.where(mutant != ref_mutant)[0])
            ref_pdf[distance] += count

        print(f'progress: {len(self.mutants)}/{len(self.mutants)}\ttime: {format(time() - t0, ".1f")} s\tpartial time: {format(time() - partial_t0, ".1f")} s')
        print(f'Total time: {format(time() - t0, ".1f")}')

        distances = np.arange(len(ref_pdf)) * self.prec
        masked_ref_pdf = ref_pdf[distances > 1.]
        masked_ref_cdf = np.array([masked_ref_pdf[:(idx + 1)].sum() for idx in range(len(masked_ref_pdf))])
        masked_ref_cdf[masked_ref_cdf == 0.] = 1.
        cd = np.log(masked_ref_cdf) / np.log(distances[distances > 1.])
        cd = np.append([0.] * len(distances[distances <= 1.]), cd)

        ref_mutant = ''.join(ref_mutant)
        scaled_distances = distances / self.site_weights.sum()
        q = abs(scaled_distances.max() - scaled_distances)
        if return_pdf:
            ref_pdf = ref_pdf / np.sum(ref_pdf)
            results = pd.DataFrame({
                'distances': scaled_distances,
                'q': q,
                'cd': cd,
                'ref_pdf': ref_pdf,
                'ref_mutant': [ref_mutant] * len(distances)
            })
        else:
            results = pd.DataFrame({
                'distances': scaled_distances,
                'q': q,
                'cd': cd,
                'ref_mutant': [ref_mutant] * len(distances)
            })
        return results



    ### Choose correct function, to avoid useless calculation
    def calculate_pdf(self, mutants):
        if self.dtype == 'Hamm' and not self.weighted: results = self._calculate_simple_pdf(mutants)
        else: results = self._calculate_pdf(mutants)
        return results

    def calculate_cd(self, mutants, ref_mutant, return_pdf = False):
        if self.dtype == 'Hamm' and not self.weighted: results = self._calculate_simple_cd(mutants, ref_mutant, return_pdf)
        else: results = self._calculate_cd(mutants, ref_mutant, return_pdf)
        return results



    ### Set modules
    def set_dtype(self, dtype : str):
        if not dtype in ("Hamm", "PAM1"): raise ValueError("Incompatible value for dtype variable. Allowed values: 'Hamm' (Hamming distance pdf), 'PAM1' (PAM1 distance pdf).")
        else:
            self.dtype = dtype
            if self.dtype == "PAM1":
                distmatrix = pd.read_csv(f'{self.inputs_d}/DistPAM1.csv')
                distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
                self.residues = tuple(distmatrix.columns)
                self.distmatrix = np.array(distmatrix)

    def set_prec(self, prec : float):
        if prec < 0.: raise ValueError("Incompatible value for prec variable. Allowed values: prec > 0.")
        elif prec == 0.: self.prec = np.min(self.site_weights)
        else: self.prec = prec

    def set_weighted(self, weighted : bool):
        self.weighted = weighted
        if self.weighted: self.site_weights = np.load(f'{self.inputs_d}/site_weights.npy')
        else: self.site_weights = np.ones(len(self.site_weights))

    def set_inputs_d(self, inputs_d):
        self.inputs_d = inputs_d
        if self.dtype == 'PAM1':
            distmatrix = pd.read_csv(f'{self.inputs_d}/DistPAM1.csv')
            distmatrix = distmatrix.drop(columns = ['Unnamed: 0'])
            self.residues = tuple(distmatrix.columns)
            self.distmatrix = np.array(distmatrix)
