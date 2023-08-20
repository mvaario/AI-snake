# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)

# for testing (test also resets)
s_testing_ai = True
s_test_rate = 250
s_test_games = 100
s_wait_time = 25

# epsilon
s_start_epsilon = 1
s_epsilon_min = 0.1
s_epsilon_decay = 0.9975

# rewards
s_penalty = 10
s_apple_score = 6
s_distance_score = 1
s_random_point = 0

# DQNAgent settings
s_game_amount = 5000
s_state_size = 32
s_deque_len = 5000
s_min_memory = 5000
s_batch_size = 5000
s_step_limit = 10

# max threads
s_threading = True
s_max_threads = 50

# training settings
s_update_rate = 10

s_load_model = False
s_load_model_name = 'Original_Model_Season_1_episode_16000.model'
s_save_model = False
s_save_rate = 1000000
s_save_model_name = 'Original_Model_Season_2'

s_episodes = 200_000
s_lr_rate = 0.001
s_discount = 0.95

# testing functional model

s_functional_model = False



