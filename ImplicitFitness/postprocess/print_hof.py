import numpy as np
import sys
import glob

from bingo.evolutionary_optimizers.evolutionary_optimizer import \
                load_evolutionary_optimizer_from_file as leoff

if __name__ == "__main__":

    DIR = sys.argv[-1]    
    FILES = glob.glob(f"{DIR}/*.pkl")
    islands = [leoff(FILE) for FILE in FILES]
    gens = [island.generational_age for island in islands]
    island = islands[np.argmax(gens)]
    hof = island.hall_of_fame

    for ind in hof:
        print(str(ind))
        p = ind.get_number_local_optimization_params()
        cs = [f"C_{i}" for i in range(p)]
        ind.set_local_optimization_params(cs)
        print()
        print(str(ind))
        print(f"Fitness: {ind.fitness}")
        print(f"Complexity: {ind.get_complexity()}")
        print(f"Stack Size: {ind.command_array.shape[0]}")

    import pdb;pdb.set_trace()

