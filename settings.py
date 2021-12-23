# game settings
size = [16, 16]
green = (0, 255, 0)
red = (0, 0, 255)

display = True
display_rate = 150
wait_time = 200

# training settings
model_name = 'conv2d_no_apple_info'
load_model_name = 'models/conv2d_no_apple_info_episode_24000_avg_0.model'
load_model = False
train = True
save_model = True
save_rate = 5000

n_episodes = 20_000
step_limit = 50
step_min = 0
# rewards
penalty = 3
apple_score = 5
distance_score = 1

# DQNAgent settings
input_shape = [11, 11]
start_epsilon = 1

deque_len = 5000
min_memory = 1000
batch_size = 128
epsilon_min = 0.01
epsilon_decay = 0.999975

lr_rate = 0.001
discount = 0.95
update_rate = 10

