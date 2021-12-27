# game settings
size = [20, 20]
green = (0, 255, 0)
red = (0, 0, 255)

display = True
display_rate = 100
wait_time = 250

# training settings
model_name = 'dense_new_info'
load_model_name = 'models/conv2d_apple_info_episode_15500_avg_5.26.model'
load_model = False
train = True
save_model = True
save_rate = 250

n_episodes = 20_000
step_limit = 50
step_min = 0
# rewards
penalty = 3
apple_score = 8
distance_score = 1


# DQNAgent settings
input_shape = [4, 8]
start_epsilon = 1

deque_len = 1000
min_memory = 200
batch_size = 64
epsilon_min = 0.01
epsilon_decay = 0.9999

lr_rate = 0.001
discount = 0.95
update_rate = 50

