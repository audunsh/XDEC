## CRINT Instructions for Abel

## Installation

### Requirements

- X-DEC (with Cryapi and Libint) 
- Crystal static binaries (crystal, properties)
- Environment variables:
> export CRYSTAL_EXE_PATH="[insert/folder/]"
>
> export CRYAPI_EXE_PATH="[insert/folder/cryapi.x]"
>
> export CRINT_PATH="[.../PeriodicDEC/utils/]"
>
- Sympy (pip3 install -user sympy)
- Python 3 (module load ...)
- Numpy (check from python terminal)

## Running on Abel

Crucial steps are in boldfont.

1. Create a project folder [project]/ containing the input file, preferrably in .d12-format.

2. **From [project]/, generate input files by executing **

> $CRINT_PATH/crint.py [input].d12 -setup 

3. **From the [project]/Crystal-folder, submit a CRYSTAL calculation for the reference state to Slurm:**

> sbatch scrint_crystal_abel.sh [input].d12

4. When job is complete, verify that fort.9 has been copied back to [project]/Crystal

5. **From [project]/, submit the wannierization routine to Slurm:**

> sbatch scrint_wann_abel.sh [input].d12

6. When job is complete, verify that the following files is present in [project]:

    - psm1.npy
    - spreads_psm1.npy
    - tail_psm1.npy
    