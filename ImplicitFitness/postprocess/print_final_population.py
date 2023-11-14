import numpy as np
import sys

from pickle_information import get_DIR_information as gdi

if __name__ == "__main__":

    DIR = sys.argv[-1]    
    if DIR[-1] == "/":
        DIR = DIR[:-1]
    info = gdi(DIR, p=0.5)
    final_pickle = info[0][-1]
    fits = [ind.fitness for ind in final_pickle.population]
    idxs = np.flip(np.argsort(fits))
    for i in idxs:
        ind = final_pickle.population[i]
        print(str(ind))
        print(f"Fitness: {ind.fitness}")
        print(f"Complexity: {ind.get_complexity()}")
        print(f"Stack Size: {ind.command_array.shape[0]}")
    import pdb;pdb.set_trace()

