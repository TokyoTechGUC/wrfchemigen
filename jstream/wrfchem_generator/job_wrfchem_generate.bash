#!/bin/bash
#$ -cwd
#$ -m abe
#$ -N jstwrfch
#$ -M alvincgvarquez@gmail.com
#$ -l h_rt=01:00:00
#$ -l cpu_40=8

set -eou pipefail
module load intel intel-mpi
module load miniconda >& /dev/null
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)" 
set +u
conda activate /gs/bs/tga-guc-lab/dependencies/dependencies_intel/conda/envs/guconda
set -u
mpirun -n 8 -ppn 1 python wrfchem_generator_jstream_parallel.py 
