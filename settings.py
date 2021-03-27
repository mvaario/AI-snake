gray = 80
green = (0, 255, 0)
red = (0, 0, 255)

size = [41, 41]

apple_score = 20
penalty = 7
move_reward = 1  # reward/penalty for getting closer to apple

load_model = False
play = False
train = True
show = True
show_rate = 500

n_episodes = 150_000
step_limit = 150
train_step = 0

# DQNA
lr_rate = 0.001
epsilon_min = 0.01
epsilon_decay = 0.99975
start_epsilon = 1


deque_len = 2000
min_memory = 1000
batch_size = 512
discount = 0.95
update_rate = 15

ep_rewards = []
save_rate = 1000
model_name = 'new'
load_model_name = 'models/new_e_118_000_avg_-10.model'