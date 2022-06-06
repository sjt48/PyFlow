#!/bin/bash
 
#SBATCH --job-name=mHBD_12                        # Job name, will show up in squeue output
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20                        # Number of cores
##SBATCH --nodes=1                                # Ensure that all cores are on one machine
#SBATCH --time=14-00:00:00                        # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=2000                        # Memory per cpu in MB (see also --mem) 
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=steven.thomson@fu-berlin.de   # Email to which notifications will be sent 
#SBATCH --array=1-10:1
# Run script
bash
#python main.py 48 'linear'
#python main.py 24 'curved'
#python main_multi.py 10 random vec

#for d in {1..10..05}
#do
#    srun --exclusive --ntasks=5 python main_multi.py 2 random vec $d &
#done
#wait

srun python main_multi.py 12 random ${SLURM_ARRAY_TASK_ID}
sleep 10

#echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
#sleep 10
