import numpy as np
from collections import defaultdict
from numba import njit
from Plague.DataScrap.LoadData import load_insect_data

data_filepath = '../DataScrap/Data/InsectData_143.csv'
data_arrays = load_insect_data(file_path=data_filepath, from_excel=False, shift_time=7, min_day=0)

T = data_arrays['t_diadegma']
W = data_arrays['Tr_diadegma']

# Aggregate sums
time_population = defaultdict(int)
for t, w in zip(T, W):
    time_population[t] += w

# Convert to arrays for Numba compatibility
lookup_times = np.array(sorted(time_population.keys()))
lookup_values = np.array([time_population[t] for t in lookup_times])

@njit
def diadegma_lookup(t):

    #Stair Lookup table
    t = int(t)  # Truncate t to an integer to return W for any int(t)
    for i in range(len(lookup_times)):
        if lookup_times[i] == t:
            return lookup_values[i]
    return 0
