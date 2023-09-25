import numpy as np
import glob

from bingo.evolutionary_optimizers.evolutionary_optimizer import \
                load_evolutionary_optimizer_from_file as leoff


def get_DIR_information(DIR, p=1):

    FILES = glob.glob(DIR+"/*.pkl")
    pickles = [leoff(FILE) for FILE in FILES]
    gens = np.array([pickle.generational_age for pickle in pickles])
    sort_idxs = np.argsort(gens)

    pickles = [pickles[i] for i in sort_idxs]
    gens.sort()
    fits = np.array([[i.fitness for i in p.population] \
                                        for p in pickles])
    comps = np.array([[i.get_complexity() for i in p.population] \
                                        for p in pickles])
    
    non_nan_idx = get_non_nan_fits_idx(fits, p)
    max_idx = gens.shape[0]
    return pickles, gens, fits, comps, max_idx, non_nan_idx


def get_non_nan_fits_idx(fits, p):

    percents = 1 - (np.isnan(fits).sum(axis=1) / fits.shape[1])
    non_nan_fits_idx = np.argmax(percents-p)

    return non_nan_fits_idx
