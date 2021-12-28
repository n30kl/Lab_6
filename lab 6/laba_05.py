import pandas as pd
import numpy as np

data = pd.read_csv("result.csv")

mathematical_expectation = sum(data['time'])/len(data['time'])
print(f'Mathematical Expectation: {mathematical_expectation}')


array_of_repetitions = np.unique(data['score'], return_counts = True)
x, n = [], [] 
for i in array_of_repetitions[0]:
    x.append(int(i))

for i in array_of_repetitions[1]:
    n.append(i)

help_arr = 0
for elem in range(len(x)):
    help_arr += x[elem] * n[elem]

mathematical_expectation_for_score = help_arr / len(data['score'])
help_arr_for_disper = 0

for elem in range(len(x)):
    help_arr_for_disper += (((x[elem] - mathematical_expectation_for_score)) ** 2) * n[elem]

disper = help_arr_for_disper / len(data['score'])
print(f'Dispersion: {disper}')


# (ql:quickload :cl-csv)