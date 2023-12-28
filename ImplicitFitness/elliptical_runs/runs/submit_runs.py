import os
import glob


if __name__ == "__main__":

    DIRS = glob.glob("./run*")
    HOME = os.getcwd()

    for DIR in DIRS:

        tag = f"ell_{DIR.split('/')[-1]}"
        os.chdir(DIR)
        os.system(f"sbatch -J {tag} ../elliptical_base.slurm")
        os.chdir(HOME)

