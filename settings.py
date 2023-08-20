# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)

# for testing (test also resets)
s_testing_ai = True
s_test_rate = 100
s_test_games = 100
s_wait_time = 25

# epsilon
s_start_epsilon = 1
s_epsilon_min = 0.1
s_epsilon_decay = 0.9975
s_step_limit = 10

# rewards
s_penalty = 10
s_apple_score = 6
s_distance_score = 1
# random point = increase snake randomly, doesn't affect reward
s_random_point = 0

# DQNAgent settings
s_state_size = 20
# game amount = memory len = batch size
s_game_amount = 250
s_update_rate = 10

s_load_model = False
s_load_model_name = 'Original_Model_Season_1_episode_16000.model'
s_save_model = True
s_save_rate = 1000
s_save_model_name = 'Functional_model'

s_episodes = 10_000
s_lr_rate = 0.001
s_discount = 0.95





