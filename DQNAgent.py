import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import numpy as np
from settings import *
from tensorflow.keras.callbacks import TensorBoard
import time
import webbrowser
import os


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, model_name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()

class DQNAgent:
    def __init__(DQNA, input_shape):
        DQNA.state_size = input_shape.shape
        DQNA.action_size = 3
        DQNA.epsilon = start_epsilon

        # Main model
        DQNA.model = DQNA.create_model_v2()

        # Target network
        DQNA.target_model = DQNA.create_model_v2()
        DQNA.target_model.set_weights(DQNA.model.get_weights())

        DQNA.replay_memory = deque(maxlen=deque_len)

        # Custom tensorboard object
        DQNA.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

    # conv2D model
    def create_model(DQNA):
        if load_model:
            model = keras.models.load_model(load_model_name)
            print("")
            print("Loaded model", load_model_name)
            print("")
        else:
            model = keras.Sequential([
                keras.layers.Conv2D(128, (2, 2), input_shape=DQNA.state_size, activation=tf.nn.relu),
                keras.layers.MaxPool2D(2, 2),
                keras.layers.Dropout(0.2),

                keras.layers.Conv2D(128, (2, 2), activation=tf.nn.relu),
                keras.layers.MaxPool2D(2, 2),
                keras.layers.Dropout(0.2),

                keras.layers.Flatten(),
                # keras.layers.Dense(64, activation=tf.nn.relu),
                keras.layers.Dense(64),
                keras.layers.Dense(DQNA.action_size, activation='linear')
            ])

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_rate),
                          loss='mse',
                          metrics=['accuracy']
                          )
            print("wrong place")
            exit()
        return model

    # test model dense
    def create_model_v2(DQNA):
        if load_model:
            model = keras.models.load_model(load_model_name)
            print("Loaded model", load_model_name)
        else:
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=DQNA.state_size),

                keras.layers.Dense(32, activation=tf.nn.relu),

                keras.layers.Dense(32, activation=tf.nn.relu),

                keras.layers.Dense(16, activation=tf.nn.relu),

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
        if logging:
            DQNA.model.fit(
                np.array(x),
                np.array(y),
                batch_size=batch_size,
                verbose=0,
                shuffle=False,
                callbacks=[DQNA.tensorboard]
            )
        else:
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

