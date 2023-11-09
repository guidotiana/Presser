import numpy as np

with open('plddt_profile.txt', 'r') as f:
    lines = f.readlines()
plddt_lines = np.array([line.split('\t') for line in lines], dtype = float)

with open('../rescaling/constants.txt', 'r') as f:
    lines = f.readlines()
constants = np.array([line.split('\t')[3] for line in lines], dtype = float)

plddt_lines[:, 0] = plddt_lines[:, 0] * constants
with open('rescaled_plddt_profile.txt', 'w') as f:
    for line in plddt_lines:
        print(f'{format(line[0], ".4f")}\t{format(line[1], ".10f")}\t{format(line[2], ".10f")}', file = f)
