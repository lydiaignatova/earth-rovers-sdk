python ~/repos/multinav-rl/liren/robot/actor_frodobot.py \
	--robot_ip localhost \
	--obs_type generic \
	--robot_type frodobot \
	--data_save_location none \
	--action_type gc_cql_local \
	--checkpoint_load_dir /home/lydia/data/checkpoints/cql_models/gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54 \
	--checkpoint_load_step 495000 \
	--deterministic \
	--goal_npz ~/data/frodobot/goals/outside1.npz \
	--step_by_one \
	--manually_advance 
