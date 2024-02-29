from settings import *
from tensorflow import keras
from collections import deque
import numpy as np
import random
# from tensorflow.python.keras.callbacks import TensorBoard
import time

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

        # DQNA.replay_memory = []
        DQNA.replay_memory = deque(maxlen=s_deque_memory)

    def create_functional_model(DQNA):
        # apple = apple and head
        snake_body = s_state_size - 4
        # split x/y and add head position
        snake_x = int(snake_body / 2 + 1)
        snake_y = int(snake_body / 2 + 1)

        # apple input
        input_head_apple = keras.Input(4, name='head_apple')
        head_apple = keras.layers.Flatten(name='head_apple_flatten')(input_head_apple)
        head_apple = keras.layers.Dense(32, activation='relu')(head_apple)

        if snake_body > 0:
            input_snake_x = keras.Input(snake_x, name='snake_x')
            snake_x = keras.layers.Flatten(name='snake_x_flatten')(input_snake_x)
            snake_x = keras.layers.Dense(64, activation='relu')(snake_x)
            snake_x = keras.layers.Dense(64, activation='relu')(snake_x)


            input_snake_y = keras.Input(snake_y, name='snake_y')
            snake_y = keras.layers.Flatten(name='snake_y_flatten')(input_snake_y)
            snake_y = keras.layers.Dense(64, activation='relu')(snake_y)
            snake_y = keras.layers.Dense(64, activation='relu')(snake_y)

            snake = keras.layers.Concatenate(name='snake')([head_apple, snake_x, snake_y])

            output = keras.layers.Dense(DQNA.action_size, activation='linear')(snake)
            model = keras.Model(inputs=[input_head_apple, input_snake_x, input_snake_y], outputs=output)
        else:
            output = keras.layers.Dense(DQNA.action_size, activation='linear')(head_apple)
            model = keras.Model(inputs=input_head_apple, outputs=output)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=s_lr_rate),
                      loss='mse',
                      metrics=['accuracy']
                      )
        return model

    def create_sequential_model(DQNA):
        # apple = apple and head
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=DQNA.state_size),

            keras.layers.Dense(128, activation='relu'),

            keras.layers.Dense(128, activation='relu'),

            keras.layers.Dense(64, activation='relu'),

            keras.layers.Dense(DQNA.action_size, activation='linear')

        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=s_lr_rate),
                      loss='mse',
                      metrics=['accuracy']
                      )
        return model

    def create_model(DQNA):
        if s_load_model:
            model = keras.models.load_model(f'D:\Programs\Coding\Projects\AI_snake\models\{s_load_model_name}')
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

    def train_model(DQNA, e):
        if len(DQNA.replay_memory) < s_deque_memory:
            print("ERROR in DQNA.replay_memory size")
            quit()
        if not s_train_model:
            return

        # train specific model
        if s_functional_model:
            DQNA.train_functional_model()
        else:
            DQNA.train_sequential_model()

        # empty memory?
        DQNA.replay_memory = deque(maxlen=s_deque_memory)
        # model modifications
        DQNA.epsilon_decay()
        DQNA.target_update(e)
        DQNA.save_model(e, force=False)

        return

    def train_sequential_model(DQNA):
        # random batch
        replay_memory = DQNA.replay_memory

        # choice current states
        current_states = np.array([transition[0] for transition in replay_memory])

        # choice the new states (change type to array -> much faster)
        new_current_states = np.array([transition[3] for transition in replay_memory])

        # get the q values
        current_qs_list = DQNA.model(current_states, training=False)
        current_qs_list = np.array(current_qs_list)

        # get the q values (change type to array -> much faster)
        future_qs_list = DQNA.target_model(new_current_states, training=False)
        future_qs_list = np.array(future_qs_list)

        y = []
        for index, (current_state, action, step_reward, new_current_state, done) in enumerate(replay_memory):
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
            if np.all(current_state == new_current_state):
                print("Same states", current_state)
                quit()
            if np.all(current_states[index] == new_current_states[index]):
                print("Same states 2", current_state)
                quit()
            if np.all(current_state[2:] == 0):
                print(current_state)
                quit()
            if np.all(new_current_state[2:] == 0) and not done:
                print("new current state error")
                quit()


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
        # random batch
        # minibatch = random.sample(DQNA.replay_memory, s_batch_size)
        replay_memory = DQNA.replay_memory

        # choice current states
        current_states = np.array([transition[0] for transition in replay_memory])
        # choice the new states
        new_current_states = np.array([transition[3] for transition in replay_memory])

        head_apple = []
        f_head_apple = []

        snake_x = []
        snake_y = []
        f_snake_x = []
        f_snake_y = []
        for i in range(len(replay_memory)):
            head_apple = np.append(head_apple, current_states[i, 0:4])
            f_head_apple = np.append(f_head_apple, new_current_states[i, 0:4])

            if s_state_size > 4:
                current_state = current_states[i, 2:]
                new_current_state = new_current_states[i, 2:]
                # split x and y
                size = int(len(current_state) / 2)
                for k in range(size):
                    j = k * 2
                    # current state
                    snake_x = np.append(snake_x, current_state[j + 1])
                    snake_y = np.append(snake_y, current_state[j])

                    # future state
                    f_snake_x = np.append(f_snake_x, new_current_state[j + 1])
                    f_snake_y = np.append(f_snake_y, new_current_state[j])

                snake_x = np.reshape(snake_x, (-1, size))
                f_snake_x = np.reshape(f_snake_x, (-1, size))

                snake_y = np.reshape(snake_y, (-1, size))
                f_snake_y = np.reshape(f_snake_y, (-1, size))

        head_apple = np.reshape(head_apple, (-1, 4))
        f_head_apple = np.reshape(f_head_apple, (-1, 4))

        if s_state_size > 4:
            # get the q values
            current_qs_list = DQNA.model((head_apple, snake_x, snake_y), training=False)
            current_qs_list = np.array(current_qs_list)

            # get the q values
            future_qs_list = DQNA.target_model((f_head_apple, f_snake_x, f_snake_y), training=False)
            future_qs_list = np.array(future_qs_list)
        else:
            # get the q values
            current_qs_list = DQNA.model(head_apple, training=False)
            current_qs_list = np.array(current_qs_list)

            # get the q values
            future_qs_list = DQNA.target_model(f_head_apple, training=False)
            future_qs_list = np.array(future_qs_list)

        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(replay_memory):
            if not done:
                # calculate max reward
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + s_discount * max_future_q
            else:
                new_q = reward

            # get the q values
            current_qs = current_qs_list[index]

            # change the action q value to the reward
            current_qs[action] = new_q

            y.append(current_qs)

            # check data for no reason at all
            if np.all(current_state == new_current_state):
                print("Same states", current_state)
                quit()
            if np.all(current_states[index] == new_current_states[index]):
                print("Same states 2", current_state)
                quit()
            if np.all(current_state[2:] == 0):
                print(current_state)
                quit()
            if np.all(new_current_state[2:] == 0) and not done:
                print("new current state error")
                quit()

        # fit model to the rewards
        DQNA.model.fit(
            (head_apple, snake_x, snake_y),
            np.array(y),
            batch_size=s_batch_size,
            verbose=0,
            shuffle=True,
            # callbacks=[tensorboard],
            epochs=s_epochs
        )

        return

    # pick action
    def get_qs(DQNA, state, r_testing):
        if r_testing or np.random.rand() > DQNA.epsilon:
            state = np.reshape(state, (-1, s_state_size))
            if s_functional_model:
                head_apple = state[:, 0:4]
                snake = state[:, 2:]
                snake = snake[0]
                snake_x = []
                snake_y = []
                size = int(len(snake) / 2)
                for k in range(size):
                    j = k * 2
                    # current state
                    snake_x = np.append(snake_x, snake[j + 1])
                    snake_y = np.append(snake_y, snake[j])

                snake_x = np.reshape(snake_x, (-1, size))
                snake_y = np.reshape(snake_y, (-1, size))

                if s_state_size > 4:
                    act_values = DQNA.model((head_apple, snake_x, snake_y), training=False)
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
            DQNA.model.save(f'D:\Programs\Coding\Projects\AI_snake\models\{s_save_model_name}_episodes_{e}.model')
            time.sleep(0.1)
        return