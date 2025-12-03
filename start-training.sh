#!/bin/bash

# Function to check if a tmux session exists
session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first."
    exit 1
fi

echo "Starting tmrl components in separate tmux sessions..."

# 1. Launch Server
SESSION_NAME="tmrl_server"
if session_exists "$SESSION_NAME"; then
    echo "Session '$SESSION_NAME' already exists. Skipping."
else
    echo "Launching Server in session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "python -m tmrl --server" C-m
fi

# 2. Launch Trainer (with wandb)
SESSION_NAME="tmrl_trainer"
if session_exists "$SESSION_NAME"; then
    echo "Session '$SESSION_NAME' already exists. Skipping."
else
    echo "Launching Trainer in session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "python -m tmrl --trainer --wandb" C-m
fi

# 3. Launch Worker
SESSION_NAME="tmrl_worker"
if session_exists "$SESSION_NAME"; then
    echo "Session '$SESSION_NAME' already exists. Skipping."
else
    echo "Launching Worker in session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME"
    tmux send-keys -t "$SESSION_NAME" "python -m tmrl --worker" C-m
fi

echo "All commands launched!"
echo "Use 'tmux list-sessions' to view them."
echo "Use 'tmux attach -t <session_name>' to view a specific session."
