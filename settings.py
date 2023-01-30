# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)

# rates
s_update_rate = 10
s_save_rate = 500

# for testing (test also resets)
s_testing_ai = True
s_test_rate = 50
s_test_games = 10
s_wait_time = 100

# epsilon
s_start_epsilon = 1
s_epsilon_min = 0.05
s_epsilon_decay = 0.9975

# rewards
s_penalty = 5
s_apple_score = 5
s_distance_score = 0.5

# DQNAgent settings
s_state_size = 128
s_deque_len = 7500
s_min_memory = 5000
s_game_amount = 150
s_batch_size = 1000

# training settings
s_model_name = 'dense_256_128_64'
s_load_model_name = 'dense_256_128_64'
s_episodes = 10_000
s_lr_rate = 0.001
s_discount = 0.95






