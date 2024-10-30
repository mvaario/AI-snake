# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)
s_control = False
s_screen_size = (350, 350)

# for testing
s_test_model = True
s_test_rate = 100
s_test_games = 10
s_loop_time = 0.02

# epsilon
s_start_epsilon = 0.75
s_epsilon_min = 0.1
s_epsilon_decay = 0.99975

# rewards
s_penalty = -35
s_apple_score = 15
s_distance_score = 1
s_distance_score_minus_multiplier = 2

# maximum snake len
s_max_len = 0
s_sort_state = False
# (state size - 4) / 2 = length


# DQNAgent settings
s_episodes = 500_000
s_state_size = (20, 20)
s_lr_rate = 0.00025
s_discount = 0.95
s_update_rate = 15
s_epochs = 1

# game amount = memory len = batch size (much be same when using multiple games)
s_game_amount = 64
s_deque_memory = 64
s_batch_size = 64

# limits
s_start_len = 0
s_start_step_limit = 10
s_step_limit_multiplier = 1
s_score_limit_multiplier = 0.7
s_allow_snake_grow = True

# level up
s_epsilon_increase = 0.2
s_step_increase = 50
s_add_len = 0

# load / save settings
s_train_model = True
s_functional_model = False
s_load_model = False
s_path = '\\192.168.1.157\\SharedFolder\\AI_Snake\\models'
s_load_model_name = 'Conv2d_256_512_512_Dense_512_256_continue3_episodes_60000.keras'
s_save_model = False

# save rate uses steps not games
s_save_rate = 10000
s_save_model_name = 'Conv2d_256_512_512_Dense_512_256_continue4'


# PPO
s_use_ppo = True

s_ppo_memory_len = 10000
s_ppo_min_memory = 5000



