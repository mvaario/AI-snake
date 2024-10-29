import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *


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

        PPO.memory = []

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

    def update_memory(PPO, state, action, step_reward, next_state, done):
        PPO.memory.append((state, action, step_reward, next_state, done))
        return

    def get_action(PPO, state, r_testing):
        # print("r_testing not in use")
        # Seems like we are shaping the state multiple times

        size = -1, PPO.state_size[0], PPO.state_size[1]
        state = np.reshape(state, size)

        action_probs = PPO.actor_network(state, training=False)

        # print("Action probabilities:", action_probs[0])
        # print("Sum of probabilities:", np.sum(action_probs[0]))

        # should fix if sum not 1
        action_probs = action_probs.numpy()

        if r_testing:
            action = np.argmax(action_probs)
        else:
            # Sample action based on probabilities
            action = np.random.choice(range(action_probs.shape[1]), p=action_probs[0])

        return action

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
        if len(PPO.memory) < s_deque_memory:
            print("ERROR in PPO.memory size", len(PPO.memory))
            return

        ratio, total_loss = PPO.train_sequential_ppo_model()

        PPO.memory = []
        PPO.save_model(e, force=False)

        return  ratio, total_loss

    def train_sequential_ppo_model(PPO):
        # Gather all experience from the replay buffer
        states, actions, rewards, next_states, dones = zip(*PPO.memory)

        # Convert gathered experience to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)  # Ensure rewards are float32
        next_states = np.array(next_states)
        dones = np.array(dones)

        # Get values and next values, compute advantages and target values
        values = tf.squeeze(PPO.critic_network(states, training=False))  # Use training=False to fix model params
        next_values = tf.squeeze(PPO.critic_network(next_states, training=False))

        advantages = tf.convert_to_tensor(rewards + PPO.gamma * next_values * (1 - dones) - values, dtype=tf.float32)
        target_values = tf.convert_to_tensor(rewards + PPO.gamma * next_values * (1 - dones), dtype=tf.float32)

        # Get log probabilities of the actions taken under the old policy
        action_probs_old = PPO.actor_network(states, training=False)  # Set training=False here
        actions_onehot = tf.one_hot(actions, PPO.action_size)  # Adjust for action space size
        log_probs_old = tf.reduce_sum(actions_onehot * tf.math.log(action_probs_old + 1e-10), axis=1)

        # Begin gradient computation and policy update
        with tf.GradientTape() as tape:
            # Get new action probabilities (these will change as we update in each batch)
            new_action_probs = PPO.actor_network(states, training=True)
            log_probs_new = tf.reduce_sum(actions_onehot * tf.math.log(new_action_probs + 1e-10), axis=1)

            # Calculate ratio and clipped ratio
            ratio = tf.exp(log_probs_new - log_probs_old)
            epsilon = 0.2  # PPO clipping range
            clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
            advantage = tf.stop_gradient(advantages)

            # Calculate PPO loss
            ppo_loss = -tf.reduce_mean(tf.minimum(ratio * advantage, clipped_ratio * advantage))

            # Calculate critic loss using updated values
            values = tf.squeeze(PPO.critic_network(states, training=True))
            critic_loss = tf.reduce_mean(tf.square(target_values - values))

            # Combine losses
            total_loss = ppo_loss + 0.5 * critic_loss

        # Compute gradients and apply
        grads = tape.gradient(total_loss,
                              PPO.actor_network.trainable_variables + PPO.critic_network.trainable_variables)

        # Split gradients for each network and apply separately
        PPO.actor_optimizer.apply_gradients(zip(grads[:len(PPO.actor_network.trainable_variables)],
                                                PPO.actor_network.trainable_variables))
        PPO.critic_optimizer.apply_gradients(zip(grads[len(PPO.actor_network.trainable_variables):],
                                                 PPO.critic_network.trainable_variables))

        return ratio, total_loss

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
