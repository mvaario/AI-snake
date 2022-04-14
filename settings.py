# game settings
size = [20, 20]
green = (0, 255, 0)
red = (0, 0, 255)

display = False
display_rate = 250
wait_time = 100

# training settings
model_name = 'dense_63+32+16'
load_model_name = 'models/dense_32x2_+28500_avg_40.39.model'
load_model = False
train = True
save_model = False
save_rate = 500
logging = False

n_episodes = 10_00_000
step_limit = 5000
step_min = 0

# rewards
penalty = 6
apple_score = 8
distance_score = 1

# DQNAgent settings
state_size = [7, 4]
deque_len = 500000
min_memory = 500000
batch_size = 500000

start_epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.99995

lr_rate = 0.001
discount = 0.85
update_rate = 10

