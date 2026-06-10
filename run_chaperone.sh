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
    
    # Activate the uv-managed project venv (created by scripts/setup_env.sh).
    if [ -f .venv/bin/activate ]; then
        echo "Environment: activating uv .venv..."
        source .venv/bin/activate
    else
        echo "No .venv found — run: bash scripts/setup_env.sh" >&2
        exit 1
    fi

    # Run the local Gemma backend on the allocated GPU.
    export CHAPERONE_LLM__BACKEND=gemma
    export CHAPERONE_EMBEDDING__DEVICE=cuda

    echo "Starting Chaperone (backend=$CHAPERONE_LLM__BACKEND)..."
    python main.py chat

    echo "Session ended. Relinquishing GPU node."
'