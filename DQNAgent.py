import cv2

from settings import *
from tensorflow import keras
import numpy as np
import time
import random

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

        DQNA.replay_memory = []

    # dense model
    def create_model(DQNA):
        if s_load_model:
            model = keras.models.load_model(f'D:\Programs\Coding\Projects\AI_snake\models\{s_load_model_name}')
            print("")
            print(f'Model {s_load_model_name} loaded')
        else:
            # apple = apple and head
            head_apple = 4
            snake_body = s_state_size - head_apple

            # apple input
            input_head_apple = keras.Input(head_apple, name='head_apple')
            head_apple = keras.layers.Flatten(name='Head_apple_flatten')(input_head_apple)
            head_apple = keras.layers.Dense(8, activation='relu')(head_apple)

            # snake body input
            if snake_body > 0:
                input_snake_body = keras.Input(snake_body, name='snake_body')
                snake_body = keras.layers.Flatten(name='Snake_body_flatten')(input_snake_body)
                snake_body = keras.layers.Dense(8, activation='relu')(snake_body)

                snake = keras.layers.Concatenate(name='Snake')([head_apple, snake_body])
                snake = keras.layers.Dense(16, activation='relu')(snake)
                snake = keras.layers.Dense(8, activation='relu')(snake)

                output = keras.layers.Dense(DQNA.action_size, activation='linear')(snake)
                model = keras.Model(inputs=[input_head_apple, input_snake_body], outputs=output)
            else:
                output = keras.layers.Dense(DQNA.action_size, activation='linear')(head_apple)
                model = keras.Model(inputs=input_head_apple, outputs=output)


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
        # choice current states
        current_states = np.array([transition[0] for transition in DQNA.replay_memory])
        # choice the new states
        new_current_states = np.array([transition[3] for transition in DQNA.replay_memory])

        head_apple = []
        snake_body = []
        f_head_apple = []
        f_snake_body = []
        for i in range(len(current_states)):
            head_apple.append(current_states[i, 0:4])
            snake_body.append(current_states[i, 4:])

            if s_state_size > 4:
                f_head_apple.append(new_current_states[i, 0:4])
                f_snake_body.append(new_current_states[i, 4:])

        head_apple = np.reshape(head_apple, (-1, 4))
        f_head_apple = np.reshape(f_head_apple, (-1, 4))

        if s_state_size > 4:
            snake_body = np.reshape(snake_body, (-1, s_state_size - 4))
            f_snake_body = np.reshape(f_snake_body, (-1, s_state_size - 4))

            # get the q values
            current_qs_list = DQNA.model((head_apple, snake_body), training=False)

            # get the q values
            future_qs_list = DQNA.target_model((f_head_apple, f_snake_body), training=False)
        else:
            # get the q values
            current_qs_list = DQNA.model(head_apple, training=False)

            # get the q values
            future_qs_list = DQNA.target_model(f_head_apple, training=False)

        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(DQNA.replay_memory):
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

        # reset replay memory
        DQNA.replay_memory = []

        # fit model to the rewards
        DQNA.model.fit(
            (head_apple, snake_body),
            np.array(y),
            batch_size=s_batch_size,
            verbose=False,
            shuffle=True,
            # callbacks=[tensorboard],
            epochs=1
        )

        return

    # pick action
    def get_qs(DQNA, state, r_testing):
        if r_testing or np.random.rand() > DQNA.epsilon:
            # current state
            head_apple = state[0:4]
            snake_body = state[4:]

            head_apple = np.reshape(head_apple, (-1, 4))
            if len(snake_body) != 0:
                snake_body = np.reshape(snake_body, (-1, s_state_size - 4))

            act_values = DQNA.model((head_apple, snake_body), training=False)

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