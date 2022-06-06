#!/bin/bash
 
#SBATCH --job-name=HBD_10                         # Job name, will show up in squeue output
#SBATCH --ntasks=20                                # Number of cores
#SBATCH --nodes=1                                 # Ensure that all cores are on one machine
#SBATCH --time=14-00:00:00                        # Runtime in DAYS-HH:MM:SS format
#SBATCH --mem-per-cpu=5000                        # Memory per cpu in MB (see also --mem) 
#SBATCH --mail-type=END                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=steven.thomson@fu-berlin.de   # Email to which notifications will be sent 

# Run script
bash
#python main.py 48 'linear'
#python main.py 24 'curved'
python main.py 10 random jit

