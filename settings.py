# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)

# rates
s_update_rate = 15
s_save_rate = 500

# for testing (test also resets)
s_testing_ai = True
s_test_rate = 50
s_test_games = 10
s_wait_time = 200

# epsilon
s_start_epsilon = 1
s_epsilon_min = 0.05
s_epsilon_decay = 0.9975

# rewards
s_penalty = 8
s_apple_score = 5
s_distance_score = 0.5

# DQNAgent settings
s_state_size = 128
s_deque_len = 10000
s_min_memory = 7500
s_game_amount = 750
s_batch_size = 3750

# training settings
s_model_name = 'dense_256_128_128'
s_load_model_name = 'models/dense_32x2.model'
s_episodes = 100_000
s_lr_rate = 0.001
s_discount = 0.95






