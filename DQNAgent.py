import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import numpy as np
from settings import *

class DQNAgent:
    def __init__(DQNA, input_shape):
        DQNA.state_size = input_shape.shape
        DQNA.action_size = 4
        DQNA.epsilon = start_epsilon

        # Main model
        DQNA.model = DQNA.create_model()

        # Target network
        DQNA.target_model = DQNA.create_model()
        DQNA.target_model.set_weights(DQNA.model.get_weights())

        DQNA.replay_memory = deque(maxlen=deque_len)

    def create_model(DQNA):
        if load_model or play:
            model = keras.models.load_model(load_model_name)
        elif train:
            model = keras.Sequential([
                keras.layers.Conv2D(512, (10, 10), input_shape=DQNA.state_size, activation=tf.nn.relu),
                keras.layers.MaxPool2D(2, 2),
                keras.layers.Dropout(0.25),

                keras.layers.Conv2D(512, (5, 5), activation=tf.nn.relu),
                keras.layers.MaxPool2D(2, 2),
                keras.layers.Dropout(0.17),

                keras.layers.Conv2D(256, (3, 3), activation=tf.nn.relu),
                keras.layers.MaxPool2D(2, 2),
                keras.layers.Dropout(0.12),

                keras.layers.Flatten(),
                keras.layers.Dense(128, activation=tf.nn.relu),
                keras.layers.Dense(DQNA.action_size, activation='linear')
            ])

            model.compile(optimizer=keras.optimizers.Adam(lr=lr_rate),
                          loss='mse',
                          metrics=['accuracy']
                          )
        return model

    def update_replay_memory(DQNA, state, action, reward, next_state, done):
        DQNA.replay_memory.append((state, action, reward, next_state, done))

    def train_2_model(DQNA, e):
        if len(DQNA.replay_memory) < min_memory:
            return

        minibatch = random.sample(DQNA.replay_memory, batch_size)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = DQNA.model.predict(current_states)
        # current_qs_list = DQNA.model(current_states, training=False)

        new_current_states = np.array([transition[3] for transition in minibatch])
        # future_qs_list = DQNA.target_model.predict(new_current_states)
        future_qs_list = DQNA.target_model(new_current_states, training=False)

        x = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)

        DQNA.model.fit(np.array(x), np.array(y), batch_size=batch_size, verbose=0, shuffle=False)

        if e % update_rate == 0:
            DQNA.target_model.set_weights(DQNA.model.get_weights())

    def get_qs(DQNA, state, r, step):
        if train and step > train_step:
            if np.random.rand() > DQNA.epsilon:
                act_values = DQNA.model.predict(np.array(state).reshape(-1, *state.shape))[0]
                action = np.argmax(act_values)
            else:
                action = np.random.randint(DQNA.action_size)
                # if DQNA.epsilon > 0.7:
                #     if action == 0 and np.all(r == [-1, 0]):
                #         action = 1
                #     elif action == 1 and np.all(r == [1, 0]):
                #         action = 2
                #     elif action == 2 and np.all(r == [0, 1]):
                #         action = 3
                #     elif action == 3 and np.all(r == [0, -1]):
                #         action = 0
        else:
            act_values = DQNA.model.predict(np.array(state).reshape(-1, *state.shape))[0]
            action = np.argmax(act_values)

        return action


