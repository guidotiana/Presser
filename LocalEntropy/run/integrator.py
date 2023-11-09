import numpy as np
import os
from os import listdir
from os.path import isfile, isdir


class Integrator:

    ### Initialization
    def __init__(
        self,
        inputs_dir : str,
        d_type : str = 'Hamm',
        discarded_mutations : int = 0,
        gamma_min : float = 0.,
        const_step : bool = True,
        initialize : bool = False
    ):
        if inputs_dir[-1] == '/': inputs_dir = inputs_dir[:-1]
        self.inputs_dir = inputs_dir
        
        assert d_type in ('PAM1', 'Hamm'), 'Invalid value for d_type parameter. Allowed values: d_type = "PAM1" or d_type = "Hamm".'
        self.d_type = d_type
        if self.d_type == 'PAM1': self.d_idx = 3
        else: self.d_idx = 4

        assert discarded_mutations >= 0, 'Invalid value for discarded_mutations parameter. Allowed values: discarded_mutations >= 0.'
        self.discarded_mutations = discarded_mutations
        self.gamma_min = gamma_min
        self.const_step = const_step
        
        self._get_id()
        self._check_directory()
        
        if initialize: self.initialize()
        


    ### Prepare data file id
    def _get_id(self):
        splitted_inputs_dir = self.inputs_dir.split('/')
        MM_dir, seed_dir = splitted_inputs_dir[-2], splitted_inputs_dir[-1]
        MM_dir_list = MM_dir.split('_')[1:3]
        self.file_id = f'{MM_dir_list[0]}_{MM_dir_list[1]}_{seed_dir}'

        

    ### Check for directory to store integration results (derived from inputs_dir)
    def _check_directory(self):
        self.results_dir = f'../results/{self.file_id}'
        
        path = self.results_dir.split('/')[1:]
        actual_dir = '..'
        for idx, new_dir in enumerate(path):
            if idx > 0:
                actual_dir = actual_dir + '/' + path[idx - 1]
            onlydirs = [d for d in listdir(f'{actual_dir}') if isdir(f'{actual_dir}/{d}')]
            if (new_dir in onlydirs) == False:
                os.mkdir(f'{actual_dir}/{new_dir}')



    ### Prepare for integration
    def initialize(self):
        self.load_data(self.const_step)
        self.calculate_means()



    ### Load data from inputs_dir
    def load_data(self, const_step):
        # Load dlists and gamma
        gammas, dlists = [], []
        data_files = [filename for filename in listdir(f'{self.inputs_dir}') if isfile(f'{self.inputs_dir}/{filename}') and ('data' in filename) and (not 'eq' in filename)]
        
        for data_file in data_files:
            with open(f'{self.inputs_dir}/{data_file}', 'r') as file:
                lines = file.readlines()
            length = float(lines[0].split('\t')[-1])
            gamma = float(lines[0].split('\t')[-2])
            dlist = [float(line.split('\t')[self.d_idx]) / length for line in lines]
            if gamma >= self.gamma_min:
                gammas.append(gamma)
                dlists.append(dlist[self.discarded_mutations:])
        
        self.gammas = np.sort(gammas)[::-1]
        self.dlists = [dlists[gammas.index(gamma)] for gamma in self.gammas]

        if const_step:
            shifted_gammas = np.append(self.gammas[1:], [self.gammas[0]])
            steps = (self.gammas - shifted_gammas)[:-1]
            assert len(np.unique(steps)) == 1, "Different intervals between simulation gammas."

        # Load S0 value
        max_idx = np.argmax(gammas)
        with open(f'{self.inputs_dir}/{data_files[max_idx]}', 'r') as file:
            lines = file.readlines()
        self.U = float(lines[0].split('\t')[1])
        self.T = float(lines[0].split('\t')[-3])
        self.S0 = -self.U / self.T
        


    ### Identify usefull parameters for reweighting
    def sort_by_distance(self, arr, el, num = -1):
        if num == -1: num = len(arr)
        sorted_arr = []
        while len(arr) > 0:
            diffs = abs(arr - el)
            sorted_arr.append(arr[diffs.argmin()])
            arr = np.delete(arr, diffs.argmin(), axis = 0)
        return np.sort(sorted_arr[:num])[::-1]



    ### Reweight distances list
    def reweight_list(self, dlist, old_gamma, new_gamma):
        dlist = np.array(dlist)
        denominator = np.exp(-(new_gamma - old_gamma) * dlist)
        numerator = dlist * denominator
        return np.mean(numerator) / np.mean(denominator)


    
    ### Predict mean distance
    def mean_prediction(self, new_gamma):
        pred_sum, norm = 0., 0.
        useful_gammas = self.sort_by_distance(self.gammas, new_gamma)
        for dlist, old_gamma in zip(self.dlists, self.gammas):
            if old_gamma in useful_gammas:
                weight = (1. - abs(new_gamma - old_gamma)/abs(new_gamma + old_gamma)) ** 10.
                pred_sum += weight * self.reweight_list(dlist, old_gamma, new_gamma)
                norm += weight
        return pred_sum / norm


    
    ### Calculate weighted means
    def calculate_means(self):
        self.means = [
                self.mean_prediction(gamma) for gamma in self.gammas
        ]
        self.means = np.array(self.means, dtype = float)



    ### Simpson method
    def Simpson(self):
        self.S = np.zeros(len(self.gammas), dtype = float)
        self.S[0] = self.S0

        for idx in range(len(self.gammas[:-1])):
            step = (self.gammas[idx] - self.gammas[idx + 1])/2.
            predicted_mean = self.mean_prediction(self.gammas[idx] + step)
            increment = (step / 3.) * (self.means[idx] + 4.*predicted_mean + self.means[idx + 1])
            self.S[idx + 1] = self.S[idx] + increment

        with open(f'{self.results_dir}/Simpson_{self.d_type}_{self.file_id}.dat', 'w') as file:
            for gamma_n, S_n, dS_n in zip(self.gammas, self.S, -self.means):
                print(f'{gamma_n}\t{S_n}\t{S_n-self.S[0]}\t{dS_n}', file = file)



    ### MidPoint integration method
    def MidPoint(self):
        self.S = np.zeros(len(self.gammas), dtype = float)
        self.S[0] = self.S0

        for idx in range(len(self.gammas[:-1])):
            step = (self.gammas[idx] - self.gammas[idx + 1])/2.
            predicted_mean = self.mean_prediction(self.gammas[idx] + step)
            increment = 2. * step * predicted_mean
            self.S[idx + 1] = self.S[idx] + increment

        with open(f'{self.results_dir}/MidPoint_{self.d_type}_{self.file_id}.dat', 'w') as file:
            for gamma_n, S_n, dS_n in zip(self.gammas, self.S, -self.means):
                print(f'{gamma_n}\t{S_n}\t{S_n-self.S[0]}\t{dS_n}', file = file)



    ### Set modules
    def set_d_type(self, d_type : str):
        assert d_type in ('PAM1', 'Hamm'), 'Invalid value for d_type parameter. Allowed values: d_type = "PAM1" or d_type = "Hamm".'
        self.d_type = d_type
        if self.d_type == 'PAM1': self.d_idx = 3
        else: self.d_idx = 4
        self.initialize()

    def set_inputs_dir(self, inputs_dir : str):
        if inputs_dir[-1] == '/': inputs_dir = inputs_dir[:-1]
        self.inputs_dir = inputs_dir
        self._get_id()
        self._check_dir()

    def set_const_step(self, const_step : bool):
        self.const_step = const_step
        self.initialize()

    def set_gamma_min(self, gamma_min : float):
        self.gamma_min = gamma_min
        self.initialize()
