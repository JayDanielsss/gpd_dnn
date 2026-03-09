import pandas as pd

numpystuff = pd.read_csv('numpy.txt', sep = ':', header = None, names = ['title', 'numbers'])
tfstuff = pd.read_csv('tensorflow.txt', sep = ':', header = None, names = ['title', 'numbers'])

residuals = numpystuff['numbers'] - tfstuff['numbers']
alarming_discrepancies = residuals[residuals > 1e-7]

print(alarming_discrepancies)