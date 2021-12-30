# game settings
size = [20, 20]
green = (0, 255, 0)
red = (0, 0, 255)

display = True
display_rate = 100
wait_time = 250

# for check terminal: tensorboard --logdir=logs/
# training settings
model_name = 'new_dense_32x2_+7500.model'
load_model_name = 'models/new_dense_32x2.model_episode_7500_avg_23.8.model'
load_model = True
train = True
save_model = True
save_rate = 500
logging = True

n_episodes = 20_000
step_limit = 150
step_min = 10
# rewards
penalty = 5
apple_score = 7
distance_score = 1

# DQNAgent settings
state_size = 24
start_epsilon = 1

deque_len = 5000
min_memory = 500
batch_size = 64
epsilon_min = 0.01
epsilon_decay = 0.9975

lr_rate = 0.001
discount = 0.95
update_rate = 5

