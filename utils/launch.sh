#!/bin/bash

# set up new session 
tmux new-session -d -s frodobot
tmux split-window -h
tmux split-window -v

tmux select-pane -t frodobot:0.0
tmux split-window -v

tmux send-keys -t frodobot:0.0 'conda activate frodobots' C-m
tmux send-keys -t frodobot:0.0 'cd ~/repos/earth-rovers-sdk' C-m
tmux send-keys -t frodobot:0.0 'hypercorn main:app --reload' C-m

tmux send-keys -t frodobot:0.1 'conda activate frodobots' C-m
tmux send-keys -t frodobot:0.1 'cd ~/repos/earth-rovers-sdk/utils' C-m
tmux send-keys -t frodobot:0.1 'python frodobot_server.py' C-m

tmux select-pane -t frodobot:2
tmux send-keys -t frodobot:0.2 'conda activate frodobots' C-m
tmux send-keys -t frodobot:0.2 'cd ~/repos/earth-rovers-sdk/utils' C-m
tmux send-keys -t frodobot:0.2 'python teleop.py' C-m

tmux select-pane -t frodobot:3
tmux send-keys -t frodobot:0.3 'conda activate frodobots' C-m
tmux send-keys -t frodobot:0.3 'cd ~/repos/earth-rovers-sdk/utils' C-m
# tmux send-keys -t frodobot:03 'python recorder.py --data_save_dir /home/lydia/data/frodobot/supervised' C-m 

tmux select-pane -t frodobot:0.0
tmux attach-session -t frodobot