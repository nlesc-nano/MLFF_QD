#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=train
#SBATCH --time=1-00:00:00  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --constraint=a100-pcie
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load Anaconda3
source activate #path of the corresponding anaconda environment

CDIR=`pwd`

export SCRATCH_DIR=/scratch/$USER/wrktmp/$SLURM_JOBID

mkdir -p $SCRATCH_DIR

# Get config file from input parameter
CONFIG_FILE=$1

# Check if config file is provided
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="input.yaml"
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi

echo "Using configuration file: $CONFIG_FILE"

# Copy necessary files to SCRATCH_DIR
cp main.py $SCRATCH_DIR
cp -r utils $SCRATCH_DIR 
cp inference.py $SCRATCH_DIR
cp "$CONFIG_FILE" $SCRATCH_DIR
cp *.npz $SCRATCH_DIR
cp *.hdf5 $SCRATCH_DIR

cd $SCRATCH_DIR

# Run training
python main.py --config "$CONFIG_FILE"

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting inference."
    # Run inference
    python inference.py --config "$CONFIG_FILE"
else
    echo "Training failed. Skipping inference."
fi

# Copy outputs back to submission directory
mkdir -p "$CDIR/$SLURM_JOB_ID"
cp -r * "$CDIR/$SLURM_JOB_ID"

rm -fr "$SCRATCH_DIR"
