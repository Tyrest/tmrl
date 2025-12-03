#!/bin/bash

# List of tmrl tmux sessions
SESSIONS=("tmrl_server" "tmrl_trainer" "tmrl_worker")

echo "Stopping tmrl sessions..."

for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session"
        echo "Killed session: $session"
    else
        echo "Session not found: $session"
    fi
done

echo "All sessions stopped."