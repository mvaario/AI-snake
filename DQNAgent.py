from settings import *
from tensorflow import keras
from collections import deque
import numpy as np
import random
# from tensorflow.python.keras.callbacks import TensorBoard
import time
import os


class DQNAgent:
    def __init__(DQNA):
        DQNA.state_size = s_state_size
        DQNA.action_size = 4
        DQNA.epsilon = s_start_epsilon

        # Main model
        DQNA.model = DQNA.create_model()

        # Target network
        DQNA.target_model = DQNA.create_model()
        DQNA.target_model.set_weights(DQNA.model.get_weights())

        DQNA.replay_memory = deque(maxlen=s_deque_memory)

    # create functional model
    def create_functional_model(DQNA):
        # (16, 2)
        snake_size = (DQNA.state_size[0] - 1, DQNA.state_size[1])

        # apple input
        input_head_apple = keras.Input(shape=(2, 2), name='head_apple')
        head_apple = keras.layers.Dense(64, activation='relu')(input_head_apple)

        # snake input
        if s_state_size[0] > 2:
            input_snake = keras.Input(shape=snake_size, name='snake')

            snake = keras.layers.Dense(256, activation='relu')(input_snake)

            snake = keras.layers.Dense(256, activation='relu')(snake)

            snake = keras.layers.Dense(64, activation='relu')(snake)

            snake = keras.layers.Flatten(name='snake_flatten')(snake)
            head_apple = keras.layers.Flatten(name='head_apple_flatten')(head_apple)

            output = keras.layers.Concatenate(name='output')([head_apple, snake])

            output = keras.layers.Dense(128, activation='relu')(output)

            output = keras.layers.Dense(DQNA.action_size, activation='linear')(output)
            model = keras.Model(inputs=[input_head_apple, input_snake], outputs=output)
        else:
            # if snake len is 0
            head_apple_flatten = keras.layers.Flatten(name='head_apple_flatten')(head_apple)
            output = keras.layers.Dense(DQNA.action_size, activation='linear')(head_apple_flatten)
            model = keras.Model(inputs=input_head_apple, outputs=output)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=s_lr_rate),
                      loss='mse',
                      metrics=['accuracy']
                      )
        return model

    # create sequential model
    def create_sequential_model(DQNA):
        size = DQNA.state_size[0], DQNA.state_size[1], 1

        model = keras.Sequential([

            keras.layers.Input(shape=DQNA.state_size),

            keras.layers.Reshape(size),

            # keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
            #
            # keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            #
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #
            # keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
            #
            # keras.layers.MaxPooling2D(pool_size=(2, 2)),
            #
            # keras.layers.Flatten(),
            #
            # keras.layers.Dense(256, activation='relu'),
            #
            # keras.layers.Dense(128, activation='relu'),


            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),

            keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),

            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'),

            keras.layers.MaxPooling2D(pool_size=(2, 2)),

            keras.layers.Flatten(),

            keras.layers.Dense(512, activation='relu'),

            keras.layers.Dense(256, activation='relu'),

            keras.layers.Dense(DQNA.action_size, activation='linear'),
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=s_lr_rate),
                      loss='mse',
                      metrics=['accuracy']
                      )

        return model

    # load or create sequential / functional model
    def create_model(DQNA):
        if s_load_model:
            # try get the model from ubuntu server
            try:
                model = keras.models.load_model(f'/home/huxiez/Python/Shared/AI_Snake/models/{s_load_model_name}')
            except:
                model = keras.models.load_model(f'\\{s_windows_path}\\{s_load_model_name}')

            print("")
            print(f'Model {s_load_model_name} loaded')

        else:
            if s_functional_model:
                model = DQNA.create_functional_model()
                print('Functional model created')
            else:
                model = DQNA.create_sequential_model()
                print('Sequential model created')

        # show model
        print(model.summary())

        return model

    def update_replay_memory(DQNA, state, action, step_reward, next_state, done):
        DQNA.replay_memory.append((state, action, step_reward, next_state, done))
        return

    # train sequential / functional model
    def train_model(DQNA, e):
        if not s_train_model:
            print("Training is not enabled")
            return
        if len(DQNA.replay_memory) < s_deque_memory:
            #print("ERROR in DQNA.replay_memory size", len(DQNA.replay_memory), s_deque_memory)
            return

        # train specific model
        if s_functional_model:
            DQNA.train_functional_model()
        else:
            DQNA.train_sequential_model()

        # empty memory?
        DQNA.replay_memory = deque(maxlen=s_deque_memory)
        DQNA.target_update(e)
        DQNA.epsilon_decay()
        DQNA.save_model(e, force=False)

        return

    def train_sequential_model(DQNA):
        # test the speed of this zipper
        # states, actions, rewards, next_states, dones = zip(*DQNA.replay_memory)

        # choice current states
        current_states = np.array([transition[0] for transition in DQNA.replay_memory])

        # choice the new states (change type to array -> much faster)
        new_current_states = np.array([transition[3] for transition in DQNA.replay_memory])

        # get the q values
        current_qs_list = DQNA.model(current_states, training=False)
        current_qs_list = np.array(current_qs_list)

        # get the q values (change type to array -> much faster)
        future_qs_list = DQNA.target_model(new_current_states, training=False)
        future_qs_list = np.array(future_qs_list)

        y = []

        # print(current_states[0])
        # print("")
        # print(current_qs_list[0])
        # print("")
        # print(DQNA.replay_memory[0][4])
        # print(DQNA.replay_memory[0][1])
        # print("")
        # print(new_current_states[0])
        # quit()

        for index, (current_state, action, step_reward, new_current_state, done) in enumerate(DQNA.replay_memory):
            if not done:
                # calculate max reward
                max_future_q = np.max(future_qs_list[index])
                new_q = step_reward + s_discount * max_future_q
            else:
                new_q = step_reward

            # get the q values
            current_qs = current_qs_list[index]

            # change the action q value to the reward
            current_qs[action] = new_q

            y.append(current_qs)

            # for no reason at all
            # DQNA.check_data(current_state, new_current_state, y, step_reward, done)

        # # again pointless thing
        # if len(y) != s_deque_memory:
        #     print("Wrong length")
        #     print(len(y))
        #     quit()

        # fit model to the rewards
        DQNA.model.fit(
            current_states,
            np.array(y),
            batch_size=s_batch_size,
            verbose=0,
            shuffle=True,
            # callbacks=[tensorboard],
            epochs=s_epochs
        )

        return

    def train_functional_model(DQNA):
        # choice current states
        current_states = np.array([transition[0] for transition in DQNA.replay_memory])
        # choice the new states
        new_current_states = np.array([transition[3] for transition in DQNA.replay_memory])

        head_apple = []
        f_head_apple = []
        snake = []
        f_snake = []
        # split head_apple and snake states
        for i in range(len(DQNA.replay_memory)):
            head_apple = np.append(head_apple, current_states[i, 0:2])
            f_head_apple = np.append(f_head_apple, new_current_states[i, 0:2])

            if s_state_size[0] > 2:
                snake = np.append(snake, current_states[i, 1:])
                f_snake = np.append(f_snake, new_current_states[i, 1:])

        # reshape head_apple current and future states
        head_apple = np.reshape(head_apple, (-1, 2, 2))
        f_head_apple = np.reshape(f_head_apple, (-1, 2, 2))

        # if snake length can be more than 0
        if s_state_size[0] > 4:
            # reshape snakes current and future states
            snake = np.reshape(snake, (-1, s_state_size[0] - 1, s_state_size[1]))
            f_snake = np.reshape(f_snake, (-1, s_state_size[0] - 1, s_state_size[1]))

            # get the q values
            current_qs_list = DQNA.model((head_apple, snake), training=False)
            current_qs_list = np.array(current_qs_list)

            # get the q values
            future_qs_list = DQNA.target_model((f_head_apple, f_snake), training=False)
            future_qs_list = np.array(future_qs_list)
        else:
            # get the q values
            current_qs_list = DQNA.model(head_apple, training=False)
            current_qs_list = np.array(current_qs_list)

            # get the q values
            future_qs_list = DQNA.target_model(f_head_apple, training=False)
            future_qs_list = np.array(future_qs_list)

        y = []
        for index, (current_state, action, step_reward, new_current_state, done) in enumerate(DQNA.replay_memory):
            if not done:
                # calculate max reward
                max_future_q = np.max(future_qs_list[index])
                new_q = step_reward + s_discount * max_future_q
            else:
                new_q = step_reward

            # get the q values
            current_qs = current_qs_list[index]

            # change the action q value to the reward
            current_qs[action] = new_q

            y.append(current_qs)

            # check data for no reason at all
            DQNA.check_data()

        # fit model to the rewards
        DQNA.model.fit(
            (head_apple, snake),
            np.array(y),
            batch_size=s_batch_size,
            verbose=0,
            shuffle=True,
            # callbacks=[tensorboard],
            epochs=s_epochs
        )

        return

    # check data for no reason at all
    def check_data(self, current_state, new_current_state, y, step_reward, done):
        # check data for no reason at all
        if np.any(current_state == 0):
            print("current state includes zeros")
            print(current_state)
            quit()
        if np.any(y == 0) and not done:
            print("new current state includes zeros")
            print(new_current_state)
            quit()
        if np.all(current_state == new_current_state):
            print("Same states")
            print(current_state)
            print(new_current_state)
            quit()
        if step_reward != -s_distance_score and step_reward != s_distance_score and step_reward != s_apple_score and step_reward != s_penalty:
            print("Wrong step reward")
            print(step_reward)
            quit()

        return

    # pick action
    def get_qs(DQNA, state, r_testing):
        if r_testing or np.random.rand() > DQNA.epsilon:
            size = -1, DQNA.state_size[0], DQNA.state_size[1]
            state = np.reshape(state, size)
            if s_functional_model:
                head_apple = state[:, 0:2]
                snake = state[:, 1:]
                if s_state_size > 4:
                    act_values = DQNA.model((head_apple, snake), training=False)
                else:
                    act_values = DQNA.model(head_apple, training=False)
            else:
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

    # save model
    def save_model(DQNA, e, force):
        if s_save_model and (e % s_save_rate == 0 or force):
            try:
                DQNA.model.save(f'/home/huxiez/Python/Shared/AI_Snake/models/{s_save_model_name}_episodes_{e}.keras')
            except:
                DQNA.model.save(f'{s_path}\\{s_save_model_name}_episodes_{e}.keras')
            print("model saved")
            time.sleep(0.1)
        return
