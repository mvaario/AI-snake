# game settings
s_size = [20, 20]
s_green = (0, 255, 0)
s_red = (0, 0, 255)
s_control = False
s_screen_size = (350, 350)

# for testing
s_test_model = True
s_test_rate = 1
s_test_games = 10
s_loop_time = 0.017

# epsilon
s_start_epsilon = 0.75
s_epsilon_min = 0.1
s_epsilon_decay = 0.99975

# rewards
s_penalty = -25
s_apple_score = 15
s_length_score_multiplier = 0.05
s_distance_score = 0
s_distance_score_minus_multiplier = 1

# maximum snake len
s_max_len = 500
s_apple_amount = 2
s_sort_state = False
# (state size - 4) / 2 = length

# DQNAgent settings
s_episodes = 500_000
s_state_size = (20, 20)
s_lr_rate = 0.00025
s_discount = 0.95
s_update_rate = 15
s_epochs = 4

# game amount = memory len = batch size (much be same when using multiple games)
s_game_amount = 64
s_deque_memory = 64
s_batch_size = 64

# limits
s_start_len = 1
s_start_step_limit = 500
s_step_limit_multiplier = 1
s_score_limit_multiplier = 0.75
s_allow_snake_grow = True

# level up
s_epsilon_increase = 0
s_step_increase = 0
s_add_len = 0

# load / save settings
s_train_model = False
s_functional_model = False
s_load_model = True
s_linux_path = '/home/huxiez/Python/Shared/AI_Snake/models/PPO'
s_windows_path = '\\192.168.1.157\\SharedFolder\\AI_Snake\\models\\PPO'
s_load_model_name = 'testmodel.keras'
s_save_model = False

# save rate uses steps not games
s_save_rate = 5000
s_save_model_name = 'test_model'

# PPO
s_actor_model_name = 'actor_PPO_Conv2d_64_128_Dense_256_Continue1_episodes_70000.keras'
s_critic_model_name = 'critic_PPO_Conv2d_64_128_Dense_256_Continue1_episodes_70000.keras'

s_use_ppo = True

s_ppo_memory_len = 10000
s_ppo_min_memory = 7500



