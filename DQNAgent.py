import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import numpy as np
from settings import *
import time

class DQNAgent:
    def __init__(DQNA, input_shape):
        DQNA.state_size = input_shape.shape
        DQNA.action_size = 3
        DQNA.epsilon = start_epsilon

        # Main model
        DQNA.model = DQNA.create_model()

        # Target network
        DQNA.target_model = DQNA.create_model()
        DQNA.target_model.set_weights(DQNA.model.get_weights())

        DQNA.replay_memory = deque(maxlen=deque_len)

    # dense model
    def create_model(DQNA):
        if load_model:
            model = keras.models.load_model(load_model_name)
            print("Loaded model", load_model_name)
            print("")
        else:
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=DQNA.state_size),

                keras.layers.Dense(32, activation=tf.nn.relu),

                keras.layers.Dense(32, activation=tf.nn.relu),

                # keras.layers.Dense(16, activation=tf.nn.relu),

                keras.layers.Dense(DQNA.action_size, activation='linear')
            ])

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_rate),
                          loss='mse',
                          metrics=['accuracy']
                          )
        return model

    def update_replay_memory(DQNA, state, action, reward, next_state, done):
        DQNA.replay_memory.append((state, action, reward, next_state, done))

    def train_model(DQNA, e):
        if len(DQNA.replay_memory) < min_memory:
            return

        # random patch
        minibatch = random.sample(DQNA.replay_memory, batch_size)

        # choice current states
        current_states = np.array([transition[0] for transition in minibatch])
        # get the q values
        current_qs_list = DQNA.model(current_states, training=False)

        # choice the new states
        new_current_states = np.array([transition[3] for transition in minibatch])
        # get the q values
        future_qs_list = DQNA.target_model(new_current_states, training=False)

        x = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                # calculate max reward
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            # get the q values
            current_qs = np.array(current_qs_list[index])
            # change the action q value to the reward
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        # fit model to the rewards, with or without tensorboard
        DQNA.model.fit(
            np.array(x),
            np.array(y),
            batch_size=batch_size,
            verbose=0,
            shuffle=False
        )

        # update target model
        if e % update_rate == 0:
            DQNA.target_model.set_weights(DQNA.model.get_weights())

    def get_qs(DQNA, state):
        if not train:
            DQNA.epsilon = 0

        if np.random.rand() > DQNA.epsilon:
            # predict action
            act_values = DQNA.model.predict(np.array(state).reshape(-1, *state.shape))[0]
            action = np.argmax(act_values)
        else:
            # random action
            action = np.random.randint(DQNA.action_size)

        return action
