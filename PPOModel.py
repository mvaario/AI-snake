import time
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow import keras
from settings import *
import random


# Define a custom capped exponential decay using a lambda inside a subclass of LearningRateSchedule
class CappedExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_rate, decay_rate, min_lr):
        self.lr_rate = lr_rate
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    def __call__(self, step):
        exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.lr_rate,
            decay_steps=self.decay_rate,
            decay_rate=self.decay_rate,
            staircase=True
        )(step)
        return tf.maximum(exp_decay, self.min_lr)


# Define Actor-Critic Model
class PPOModel:
    def __init__(PPO):
        PPO.state_size = s_state_size
        PPO.action_size = 4

        PPO.gamma = s_discount
        PPO.lambd = 0.95

        # learning rate
        PPO.decay_steps = 5000
        PPO.decay_rate = 0.75
        PPO.min_lr = 0.00001
        PPO.lr_rate = s_lr_rate

        PPO.memory = deque(maxlen=s_ppo_memory_len)

        PPO.actor_network = PPO.create_actor_network()  # policy  network
        PPO.critic_network = PPO.create_critic_network()  # value network

        # learning rate scheduler
        PPO.lr_schedule = CappedExponentialDecay(PPO.lr_rate, PPO.decay_rate, PPO.min_lr)

        # Define the optimizers for both networks
        PPO.actor_optimizer = keras.optimizers.Adam(learning_rate=PPO.lr_schedule)
        PPO.critic_optimizer = keras.optimizers.Adam(learning_rate=PPO.lr_schedule)

    def create_actor_network(PPO):
        if s_load_model:
            # try get the model from ubuntu server
            try:
                model = keras.models.load_model(f'{s_linux_path}/{s_actor_model_name}')
            except:
                model = keras.models.load_model(f'\\{s_windows_path}\\{s_actor_model_name}')

            print("")
            print(f'Model {s_load_model_name} loaded')
        else:
            model = keras.Sequential([
                keras.layers.Input(shape=PPO.state_size),

                keras.layers.Reshape((PPO.state_size[0], PPO.state_size[1], 1)),

                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

                # keras.layers.MaxPooling2D(pool_size=(2, 2)),

                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

                keras.layers.Flatten(),

                keras.layers.Dense(256, activation='relu'),

                keras.layers.Dense(PPO.action_size, activation='softmax')
            ])

        # show model
        print(model.summary())
        return model

    def create_critic_network(PPO):
        if s_load_model:
            # try get the model from ubuntu server
            try:
                model = keras.models.load_model(f'{s_linux_path}/{s_critic_model_name}')
            except:
                model = keras.models.load_model(f'\\{s_windows_path}\\{s_critic_model_name}')

            print("")
            print(f'Model {s_load_model_name} loaded')
        else:
            model = keras.Sequential([
                keras.layers.Input(shape=PPO.state_size),

                keras.layers.Reshape((PPO.state_size[0], PPO.state_size[1], 1)),

                keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),

                # keras.layers.MaxPooling2D(pool_size=(2, 2)),

                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

                keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),

                keras.layers.Flatten(),

                keras.layers.Dense(256, activation='relu'),

                keras.layers.Dense(1)
            ])

        # show model
        print(model.summary())
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

    def normalize_rewards(PPO, rewards, epsilon=1e-8):
        mean = np.mean(rewards)
        std = np.std(rewards)
        normalized_rewards = (rewards - mean) / (std + epsilon)
        return normalized_rewards

    def train_model(PPO, e):
        if not s_train_model:
            print("Training is not enabled")
            return s_lr_rate
        if len(PPO.memory) < s_ppo_min_memory:
            print("ERROR in PPO.memory size", len(PPO.memory))
            return s_lr_rate

        PPO.train_sequential_ppo_model()

        # don't clear memory
        # PPO.memory = []
        PPO.save_model(e, force=False)
        PPO.lr_schedule(e)

        # get current learning rate
        current_step = int(PPO.actor_optimizer.iterations)
        current_lr = PPO.lr_schedule(current_step).numpy()

        return current_lr

    def train_sequential_ppo_model(PPO):
        for _ in range(s_epochs):
            batch = random.sample(PPO.memory, s_batch_size)
            # batch = PPO.memory

            # Gather all experience from the replay buffer
            states, actions, log_probs_old, values, rewards, done = zip(*batch)

            # Convert gathered experience to numpy arrays
            states = np.array(states)
            actions = np.array(actions)
            log_probs_old = np.array(log_probs_old)
            values = np.array(values)
            rewards = np.array(rewards)
            done = np.array(done)

            rewards = PPO.normalize_rewards(rewards)
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
                epsilon = 0.15  # PPO clipping range
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

        return

    # save model
    def save_model(PPO, e, force):
        if not s_save_model:
            print("Error: Save model not enabled")
        elif e % s_save_rate == 0 or force:
            try:
                PPO.actor_network.save(
                    f'{s_linux_path}/actor_{s_save_model_name}_episodes_{e}.keras')
                PPO.critic_network.save(
                    f'{s_linux_path}/critic_{s_save_model_name}_episodes_{e}.keras')
            except:
                PPO.actor_network.save(f'{s_windows_path}\\actor_{s_save_model_name}_episodes_{e}.keras')
                PPO.critic_network.save(f'{s_windows_path}\\critic_{s_save_model_name}_episodes_{e}.keras')

            print("model saved")
            time.sleep(0.1)
        return
