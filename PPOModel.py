import time
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *
import random


# Define Actor-Critic Model
class PPOModel:
    def __init__(PPO):
        PPO.state_size = s_state_size
        PPO.action_size = 4

        PPO.gamma = s_discount
        PPO.lambd = 0.95
        PPO.decay_steps = 10000
        PPO.decay_rate = 0.96
        PPO.epsilon = s_start_epsilon

        PPO.memory = deque(maxlen=s_ppo_memory_len)

        PPO.actor_network = PPO.create_actor_network()  # policy  network
        PPO.critic_network = PPO.create_critic_network()  # value network

        # Define the optimizers for both networks
        PPO.actor_optimizer = keras.optimizers.Adam(learning_rate=s_lr_rate)
        PPO.critic_optimizer = keras.optimizers.Adam(learning_rate=s_lr_rate)

        # PPO.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(PPO.lr_rate,
        #                                                                  PPO.decay_steps,
        #                                                                  PPO.decay_rate,
        #                                                                  staircase=True
        #                                                                  )

    def create_actor_network(PPO):
        model = keras.Sequential([
            keras.layers.Input(shape=PPO.state_size),

            keras.layers.Reshape((PPO.state_size[0], PPO.state_size[1], 1)),

            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),

            # keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

            # keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),

            keras.layers.Dense(8, activation='relu'),

            keras.layers.Dense(PPO.action_size, activation='softmax')
        ])

        return model

    def create_critic_network(PPO):
        model = keras.Sequential([
            keras.layers.Input(shape=PPO.state_size),

            keras.layers.Reshape((PPO.state_size[0], PPO.state_size[1], 1)),

            keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),

            # keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

            # keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),

            keras.layers.Dense(8, activation='relu'),

            keras.layers.Dense(1)
        ])

        return model

    def update_memory(PPO, state, action, log_prob, value, step_reward, done):
        PPO.memory.append((state, action, log_prob, value, step_reward, done))
        return

    def get_action(PPO, state, r_testing):
        size = -1, PPO.state_size[0], PPO.state_size[1]
        state = np.reshape(state, size)

        action_probs = PPO.actor_network(state, training=False)
        # should fix if sum not 1
        action_probs = action_probs.numpy()

        if r_testing:
            action = np.argmax(action_probs)
            log_prob = None
            value = None
        else:
            # Sample action based on probabilities
            action = np.random.choice(range(action_probs.shape[1]), p=action_probs[0])
            # Calculate log probability of the chosen action
            action_onehot = tf.one_hot([action], PPO.action_size)
            log_prob = tf.reduce_sum(action_onehot * tf.math.log(action_probs + 1e-10), axis=1).numpy()

            value = PPO.critic_network(state, training=False).numpy()[0][0]

        return action, value, log_prob

    # Function to compute GAE
    def compute_gae(PPO, rewards, values, dones):
        # add random 0 to end of the values, this cannot be the correct way to do this
        terminal_value = tf.constant(0.0, shape=())
        values = tf.concat([values, tf.expand_dims(terminal_value, axis=0)], axis=0)

        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + PPO.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + PPO.gamma * PPO.lambd * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        return np.array(advantages)

    def train_model(PPO, e):
        if not s_train_model:
            print("Training is not enabled")
            return
        if len(PPO.memory) < s_ppo_min_memory:
            # print("ERROR in PPO.memory size", len(PPO.memory))
            return 1

        info_ratio = PPO.train_sequential_ppo_model()

        # don't clear memory
        # PPO.memory = []
        PPO.save_model(e, force=False)

        return info_ratio

    def train_sequential_ppo_model(PPO):
        batch = random.sample(PPO.memory, s_batch_size)

        # Gather all experience from the replay buffer
        states, actions, log_probs_old, values, rewards, done = zip(*batch)

        # Convert gathered experience to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        log_probs_old = np.array(log_probs_old)
        values = np.array(values)
        rewards = np.array(rewards)
        done = np.array(done)

        advantages = tf.convert_to_tensor(rewards + PPO.gamma * values * (1 - done) - values, dtype=tf.float32)

        # Begin gradient computation and policy update
        with tf.GradientTape(persistent=True) as tape:
            # Get log probabilities of the actions taken under the old policy
            actions_onehot = tf.one_hot(actions, PPO.action_size)

            # Get new action probabilities
            new_action_probs = PPO.actor_network(states, training=True)
            log_probs_new = tf.reduce_sum(actions_onehot * tf.math.log(new_action_probs + 1e-10), axis=1)

            critic_value = PPO.critic_network(states, training=True)
            critic_value = tf.squeeze(critic_value, 1)

            # Calculate ratio and clipped ratio
            ratio = tf.exp(log_probs_new - log_probs_old)

            weighted_probs = advantages * ratio
            epsilon = 0.2  # PPO clipping range
            clipped_probs = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
            weighted_clipped_probs = clipped_probs * advantages

            actor_loss = -tf.minimum(weighted_probs, weighted_clipped_probs)
            actor_loss = tf.reduce_mean(actor_loss)

            returns = advantages + values
            critic_loss = keras.losses.MSE(critic_value, returns)

        actor_params = PPO.actor_network.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_params)
        critic_param = PPO.critic_network.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_param)

        PPO.actor_optimizer.apply_gradients(zip(actor_grads, actor_params))
        PPO.critic_optimizer.apply_gradients(zip(critic_grads, critic_param))

        info_ratio = np.copy(ratio)
        info_ratio = np.sum(info_ratio)
        info_ratio = info_ratio / 64
        return info_ratio

    # save model
    def save_model(PPO, e, force):
        if s_save_model and (e % s_save_rate == 0 or force):
            try:
                PPO.actor.save(
                    f'/home/huxiez/Python/Shared/AI_Snake/models/actor_{s_save_model_name}_episodes_{e}.keras')
                PPO.critic.save(
                    f'/home/huxiez/Python/Shared/AI_Snake/models/critic_{s_save_model_name}_episodes_{e}.keras')
            except:
                PPO.actor.save(f'{s_path}\\actor_{s_save_model_name}_episodes_{e}.keras')
                PPO.critic.save(f'{s_path}\\critic_{s_save_model_name}_episodes_{e}.keras')

            print("model saved")
            time.sleep(0.1)
        return
