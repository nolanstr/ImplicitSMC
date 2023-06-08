import argparse
import numpy as np
import os

from PoroUtils import get_data, get_num_available_scenarios

def load_data(filename, num_scenarios, step_size, stress_measure,
              other_cols, plastic_threshold, two_stage, preproc_mult, 
              cherry_picked_scenarios=False, vvf_zero=False, hydro=False):
    preproc_factor = 10
    max_scenarios = get_num_available_scenarios(filename)
    if cherry_picked_scenarios:
        # EDIT here: instead of using random LCs, the cherry_picked_scenarios
        # will step through the max_scenarios every nL steps (so that it's 
        # the same LCs each time)
        scenarios = np.linspace(2, max_scenarios-1, num_scenarios, dtype = int)
        
    else:
        scenarios = np.random.choice(range(2, max_scenarios+2), num_scenarios, 
                                     replace=False)
    
    scenarios = np.sort(scenarios) # sort in ascending order
    
    if vvf_zero:
        scenarios[0] = 0
        
    if hydro: 
        scenarios[1] = 1
    data, data_names = get_data(FILENAME, scenarios,
                                stress_measure=stress_measure,
                                other_columns=other_cols,
                                plastic_strain_threshold=plastic_threshold,
                                step_size=step_size,
                                two_stage=two_stage)
    if preproc_mult:
        data[:, 3:] *= preproc_factor
    print(data_names)

    return data

if __name__ == '__main__':

    stress_space_options = {'pql': 'pql',
                            'hw': 'haigh_westergaard',
                            'inv': 'invariants',
                            '123': 'principle_stresses',
                            
                            'lode': 'lode', # add the custom stress measurements & porosity here
                            'no_lode': 'no_lode',
                            'poros': 'porosity'} 

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--stress_space", type=str,
                        help="stress measure to use",
                        choices=stress_space_options.keys(), required=True)
    parser.add_argument("-m", "--mult", action="store_true",
                        help="multiplication of peeq and vvf columns as "
                             "pre-processing step")
    parser.add_argument("--no_eps", action="store_true",
                        help="remove eps from dataset")
    parser.add_argument("--cherry_picked_scenarios", action="store_true",
                        help="use a specially selected subset of the 169 "
                             "loading scenarios")
    parser.add_argument("-r", "--req_x", type=int,
                        help="number of x values required in implicit "
                             "regression")
    parser.add_argument("-k", "--data_step", type=int, default=1,
                        help="step size in stride through data")
    parser.add_argument("--vvf_zero", action="store_true",
                        help="use the vvf = 0 load case") # I added this argument
    parser.add_argument("--hydro", action="store_true",
                        help="use the purely hydrostatic load case") # I added this argument
    args = parser.parse_args()

    # INPUT DATA PARAMS
    FILENAME = './Gurson_nL200nS500.hdf5'
    CHKPTLOADFILE = None
    N_DATA_SCENARIOS = 200
    N_DATA_STEP = args.data_step
    STRESS_SPACE = stress_space_options[args.stress_space]
    OTHER_COLS = ["VVF"]
    # OTHER_COLS = ["S_y","VVF"] # no normalize
    
    # ADD FUNCTIONALITY FOR POROSITY EVOLUTION RUNS
    if STRESS_SPACE == "porosity": # I'll use this when I do my porosity runs
        STRESS_SPACE = None # skip that portion in PoroUtils.py
        OTHER_COLS = ["VVF","del_VVF","del_PE11","del_PE22","del_PE33"] # these will be what I train on for my porosity evolution runs
    
    if not args.no_eps:
        OTHER_COLS.insert(0, "PEEQ")
    
    PE_THRESHHOLD = 0.5
    TWO_STAGE = False

    # OUTPUT PARAMS
    DIRNAME = 'pkl_files'
    CHECKPOINT_NAME = STRESS_SPACE

    # GP HYPER PARAMS
    N_GP_REPEATS = 1
    # convergence
    MAX_GENERATIONS = 50000
    ABS_TOLERANCE = 1e-9                                  
    STAGNATION_GENERATIONS = 50000
    CONVERGENCE_CHECK_FREQ = 1000
    # equations
    EQU_POP_SIZE = 128
    EQU_SIZE = 15
    OPERATORS = ["+", "-", "*",'cosh','/']
    # fitness predictors
    PRED_POP_SIZE = 32
    PRED_SIZE_RATIO = 0.05
    PRED_COMP_RATIO = 0.1
    # variation
    MUTATION_RATE = 0.75 # best performing hyperparameters
    CROSSOVER_RATE = 0.25

    data = load_data(FILENAME, N_DATA_SCENARIOS, N_DATA_STEP, STRESS_SPACE,
                     OTHER_COLS, PE_THRESHHOLD, TWO_STAGE, args.mult, 
                     args.cherry_picked_scenarios, args.vvf_zero, args.hydro)
    np.save("gurson_data", data)
