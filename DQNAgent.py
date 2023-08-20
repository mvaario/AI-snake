import time

from settings import *
import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np
import random
import threading

class DQNAgent:
    def __init__(DQNA, input_shape):
        DQNA.state_size = input_shape.shape
        DQNA.action_size = 4
        DQNA.epsilon = s_start_epsilon

        # Main model
        DQNA.model = DQNA.create_model()

        # Target network
        DQNA.target_model = DQNA.create_model()
        DQNA.target_model.set_weights(DQNA.model.get_weights())

        DQNA.replay_memory = deque(maxlen=s_deque_len)

    # dense model
    def create_model(DQNA):
        if s_load_model:
            model = keras.models.load_model(f'models/{s_load_model_name}')
            print("")
            print(f'Model {s_load_model_name} loaded')
        else:
            if not s_functional_model:
                model = keras.Sequential([
                    keras.Input(DQNA.state_size),
                    keras.layers.Flatten(),
                    keras.layers.Dense(64, activation=tf.nn.relu),
                    keras.layers.Dense(32, activation=tf.nn.relu),
                    keras.layers.Dense(16, activation=tf.nn.relu),
                    keras.layers.Dense(DQNA.action_size, activation='linear')
                ])
            # NEW FUNCTIONAL MODEL # # # # # # # # # # # # # # # # # # # # # # # # # #
            else:
                apple = 4
                snake = 28
                # apple input
                input_apple = keras.Input(apple, name='apple')
                apple = keras.layers.Flatten(name='Apple_flatten')(input_apple)
                apple = keras.layers.Dense(8, activation='relu')(apple)

                # snake input
                input_snake = keras.Input(snake, name='snake')
                snake = keras.layers.Flatten(name='Snake_flatten')(input_snake)
                snake = keras.layers.Dense(32, activation='relu')(snake)

                snake = keras.layers.Concatenate(name='Snake_body')([apple, snake])
                snake = keras.layers.Dense(16, activation='relu')(snake)

                output = keras.layers.Dense(DQNA.action_size, activation='linear')(snake)
                model = keras.Model(inputs=[input_apple, input_snake], outputs=output)
            # # # # # # # # # #  # # # # # # # # # # # # # # # # # # # # # # # # # #

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=s_lr_rate),
                          loss='mse',
                          metrics=['accuracy']
                          )
            print("")
            print('Model created')
        return model

    def update_replay_memory(DQNA, state, action, step_reward, next_state, done):
        DQNA.replay_memory.append((state, action, step_reward, next_state, done))
        return

    def train_model(DQNA):
        if len(DQNA.replay_memory) < s_min_memory:
            return

        # random batch
        minibatch = random.sample(DQNA.replay_memory, s_batch_size)

        # choice current states
        current_states = np.array([transition[0] for transition in minibatch])

        # choice the new states
        new_current_states = np.array([transition[3] for transition in minibatch])

        if not s_functional_model:
            # get the q values
            current_qs_list = DQNA.model(current_states, training=False)

            # get the q values
            future_qs_list = DQNA.target_model(new_current_states, training=False)

        # FOR NEW MODEL # # # # # # # # # # # # # # # # # # # # # # # # # #
        else:
            # current state
            apple = []
            snake = []

            # future states
            f_apple = []
            f_snake = []
            for i in range(len(current_states)):
                apple.append(current_states[i, 0:4])
                snake.append(current_states[i, 4:])

                f_apple.append(new_current_states[i, 0:4])
                f_snake.append(new_current_states[i, 4:])

            apple = np.concatenate(apple)
            snake = np.concatenate(snake)

            f_apple = np.concatenate(f_apple)
            f_snake = np.concatenate(f_snake)

            apple = np.reshape(apple, (-1, 4))
            snake = np.reshape(snake, (-1, 28))

            f_apple = np.reshape(f_apple, (-1, 4))
            f_snake = np.reshape(f_snake, (-1, 28))

            # get the q values
            current_qs_list = DQNA.model((apple, snake), training=False)

            # get the q values
            future_qs_list = DQNA.target_model((f_apple, f_snake), training=False)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                # calculate max reward
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + s_discount * max_future_q
            else:
                new_q = reward

            # get the q values
            current_qs = np.array(current_qs_list[index])

            # change the action q value to the reward
            current_qs[action] = new_q

            y.append(current_qs)

        if s_functional_model:
            current_states = (apple, snake)


        # fit model to the rewards
        DQNA.model.fit(
            current_states,
            np.array(y),
            batch_size=s_batch_size,
            verbose=0,
            shuffle=False,
            # callbacks=[tensorboard],
            epochs=15
        )

        return

    # pick action
    def get_qs(DQNA, state, r_testing):
        if r_testing or np.random.rand() > DQNA.epsilon:
            if s_functional_model:
                print("get Qs")

                # current state
                apple = state[0:4]
                snake = state[4:]

                apple = np.reshape(apple, (-1, 4))
                snake = np.reshape(snake, (-1, 28))
                print(apple)
                print(snake)
                quit()
                act_values = DQNA.model((apple, snake), training=False)

            else:
                # predict action
                state = np.reshape(state, (1, s_state_size))
                act_values = DQNA.model(state, training=False)
            action = np.argmax(act_values)

        else:
            # random action
            action = np.random.randint(DQNA.action_size)

        return action

    # lower randomness
    def epsilon_decay(DQNA):
        if DQNA.epsilon > s_epsilon_min:
            DQNA.epsilon *= s_epsilon_decay
            DQNA.epsilon = max(s_epsilon_min, DQNA.epsilon)
        return

    # update target model
    def target_update(DQNA, e):
        if e % s_update_rate == 0:
            DQNA.target_model.set_weights(DQNA.model.get_weights())
        return