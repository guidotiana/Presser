import numpy as np

print('Index:')
idx = int(input())
print()

# Load data
with open('outputs/alignement_energies.txt', 'r') as f:
    lines = f.readlines()
obs = lines[idx].split('\t')
obs[-1] = obs[-1][:-1]

# Load sequence
with open('inputs/1PGB_alignement.txt', 'r') as f:
    lines = f.readlines()
code = lines[idx].split('\t')[1][:-1]
seq = lines[idx].split('\t')[-1][:-1] 

# Check
wt_seq = lines[0][:-1]
wt_arr, arr = np.array(list(wt_seq)), np.array(list(seq))
assert len(wt_arr) == len(arr), 'Wrong length.'
assert len(np.where(wt_arr != arr)[0]) == int(len(wt_arr) * float(obs[3])), 'Wrong sequence.'

# Print
names = ['index', 'energy', 'PAM distance', 'Hamming distance', 'combination', 'combination distance']
print(f'Sequence: {seq}')
print(f'Code:     {code}')
print(f'Observables:')
for ob, name in zip(obs, names): print(f'{name}: {ob}')
print()
