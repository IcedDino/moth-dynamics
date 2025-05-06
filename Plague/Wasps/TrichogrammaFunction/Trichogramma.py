import numpy as np
from collections import defaultdict
from numba import njit
from Plague.DataScrap.LoadData import load_insect_data

data_filepath = '../DataScrap/Data/InsectData_143.csv'
data_arrays = load_insect_data(file_path=data_filepath, from_excel=False, shift_time=7, min_day=0)

T = data_arrays['t_trichogramma']
W = data_arrays['Tr_trichogramma']

# Conversion factor: square inches → individuals
conversion_factor = (200 / 342) * 6.4516  # cm² to individuals

# Aggregate sums with 3-day release
time_population = defaultdict(float)
for t, w in zip(T, W):
    converted_wasps = (w * conversion_factor) / 3  # Distribute evenly over 3 days
    for i in range(3):  # Release at t, t+1, t+2
        time_population[t + i] += converted_wasps

# Convert to arrays for Numba compatibility
lookup_times = np.array(sorted(time_population.keys()), dtype=np.int32)
lookup_values = np.array([time_population[t] for t in lookup_times], dtype=np.float64)

@njit
def trich_lookup(t):

    # Stair Lookup table
    t = int(t)  # Truncate t to an integer to return W for any int(t)
    for i in range(len(lookup_times)):
        if lookup_times[i] == t:
            return lookup_values[i]
    return 0