# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)

# for testing
s_test_model = True
s_test_rate = 1000
s_test_games = 10
s_wait_time = 15

# epsilon
s_start_epsilon = 1
s_epsilon_min = 0.1
s_epsilon_decay = 0.99975

# rewards
s_penalty = -10
s_apple_score = 4
s_distance_score = 1

# maximum snake len
s_max_len = 2000

# DQNAgent settings
s_episodes = 500_000
s_state_size = 32
s_lr_rate = 0.001
s_discount = 0.95
s_update_rate = 3
s_epochs = 3

# game amount = memory len = batch size
s_game_amount = 512
s_batch_size = 512
s_deque_memory = 512

# add snake until len is
s_point_until = 16
# add snake %
s_start_random_point = 0.1

# limits
s_start_step_limit = 10
s_score_limit_multiplier = 0.8
s_step_limit_multiplier = 0.9

# level up
s_epsilon_increase = 0.25
s_step_increase = 10
s_random_point_increase = 0.05

# load / save settings
s_functional_model = True
s_load_model = False
s_load_model_name = 'Sequential_model_128x128x64_episodes_10000.model'
s_save_model = True
# save rate uses steps not games
s_save_rate = 10000
s_save_model_name = 'Sequential_model_128x128x64V2'













































