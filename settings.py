# game settings
s_size = [5, 5]
s_green = (0, 255, 0)
s_red = (0, 0, 255)

# for testing (test also resets)
s_testing_ai = True
s_test_rate = 25
s_test_games = 10
s_wait_time = 25

# epsilon
s_start_epsilon = 1
s_epsilon_min = 0.1
s_epsilon_decay = 0.99975
s_step_limit = 10

# rewards
s_penalty = -5
s_apple_score = 10
s_distance_score = 1
# maximum snake len
s_max_len = 0
# random point = increase snake randomly, doesn't affect reward
s_random_point = 0

# DQNAgent settings
s_state_size = 8
# game amount = memory len = batch size
s_game_amount = 64
s_batch_size = 32
s_update_rate = 15

s_load_model = False
s_load_model_name = 'Functional_model_episode_6000.model'
s_save_model = False
s_save_rate = 1000
s_save_model_name = 'Functional_model'

s_episodes = 100_000
s_lr_rate = 0.001
s_discount = 0.95





