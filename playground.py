import os
import random
import numpy as np

random.seed(1)
np.random.seed(1)

root_folder = '/projects/0/prjs0905/data/LIDC'

paths = []

train = []
valid = []
test = []

for folder_path, _, file_names in os.walk(root_folder):
    rel_folder_path = os.path.relpath(folder_path,root_folder)
    depth = rel_folder_path.count(os.path.sep)
    if depth == 0 and 'LIDC-IDRI' in rel_folder_path:
        paths.append(rel_folder_path.split('/')[-1])

N = len(paths)
num_train = int(N * 0.3)
num_valid = int(N * 0.1)
num_test = int(N * 0.1)
num_none = N - num_train - num_valid - num_test

# Create an array with the specified distribution
choice = np.zeros(N, dtype=int)
choice[num_train:num_train+num_valid] = 1
choice[num_train+num_valid:num_train+num_valid+num_test] = 2
choice[num_train+num_valid+num_test:] = 3

# Shuffle the array
np.random.shuffle(choice)

for i in range(len(paths)):
    if choice[i] == 0:
        train.append(paths[i])
    elif choice[i] == 1:
        valid.append(paths[i])
    elif choice[i] == 2:
        test.append(paths[i])
    elif choice[i] == 3:
        pass


with open('train_val_txt/lidc_train.txt', 'w') as file:
    for path in train:
        file.write(path + '\n')

with open('train_val_txt/lidc_valid.txt', 'w') as file:
    for path in valid:
        file.write(path + '\n')

with open('train_val_txt/lidc_test.txt', 'w') as file:
    for path in test:
        file.write(path + '\n')

print(0)
