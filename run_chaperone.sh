#!/bin/bash

# Default SLURM resource requests (feel free to modify)
PARTITION="gpu"
GPUS="1"
MEM="64G"
CPUS="8"
TIME="02:00:00"

echo "========================================================"
echo " 🚀 Requesting Interactive GPU Node for Chaperone-RAG "
echo "========================================================"
echo " Resources: Partition=$PARTITION | GPUs=$GPUS | Mem=$MEM | CPUs=$CPUS | Time=$TIME"
echo " Waiting for allocation... (Press Ctrl+C to cancel)"

# Use srun --pty to allocate the node and start a bash session that runs our commands
srun -p "$PARTITION" \
     --gres=gpu:A5500:"$GPUS" \
     --mem="$MEM" \
     -N 1 \
     -c "$CPUS" \
     --time="$TIME" \
     --pty bash -c '
    echo "========================================================"
    echo " ✅ Node allocated: $(hostname)"
    echo "========================================================"
    
    # Initialize conda in this subshell
    eval "$(conda shell.bash hook)" 2>/dev/null || source activate chaperone_env
    
    echo "Environment: Activating chaperone_env..."
    conda activate chaperone_env
    
    echo "Ensuring kagglehub is installed for weights..."
    pip install kagglehub -q
    pip show langchain-chroma >/dev/null 2>&1 || pip install langchain-chroma -q

    # Keep generation latency bounded for interactive use.
    export CHAPERONE_MAX_NEW_TOKENS=1024
    export CHAPERONE_MAX_GENERATION_TIME=60
    export USER_AGENT="Chaperone-RAG/0.1 (+HPC interactive session)"

    # Add this line to force Gemma to generate all chat responses
    export CHAPERONE_CHAT_LLM=1
    
    echo "Starting Chaperone Agent..."
    python main.py
    
    echo "Session ended. Relinquishing GPU node."
'
