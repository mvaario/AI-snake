# game settings
size = [20, 20]
green = (0, 255, 0)
red = (0, 0, 255)

display = True
display_rate = 200
wait_time = 100

# for check terminal: "tensorboard --logdir=logs/
# training settings
model_name = 'dense_32x2+16_3x8.model'
load_model_name = 'models/dense_new_info+4000_episode_4500_avg_25.31.model'
load_model = False
train = True
save_model = True
save_rate = 500
logging = True

n_episodes = 10_000
step_limit = 50
step_min = 0
# rewards
penalty = 3
apple_score = 8
distance_score = 1

# DQNAgent settings
input_shape = [3, 8]
start_epsilon = 1

deque_len = 1000
min_memory = 200
batch_size = 64
epsilon_min = 0.01
epsilon_decay = 0.99975

lr_rate = 0.001
discount = 0.95
update_rate = 10

