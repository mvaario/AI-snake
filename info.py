import time

import keras.backend

from settings import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


class info:
    def __init__(self, tf):
        self.avg_scores = []
        self.avg_step = []
        self.test_count = []
        self.tensorflow_setups(tf)

    def tensorflow_setups(self, tf):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        # check if tensorflow uses GPU or CPU
        print("")
        print("")
        if len(tf.config.list_physical_devices('GPU')) == 1:
            print("Tensorflow using GPU")
        else:
            print("Tensorflow using CPU")

        # Random setup prints
        print(f'Game amount: {s_game_amount}')

        if s_save_model:
            print(f'Model save rate: {s_save_rate} with name: {s_save_model_name}')
        else:
            print("Saving is OFF")
        if s_testing_ai:
            print(f'Test rate: {s_test_rate}')
        else:
            print("Testing is OFF")
        print("")
        return

    # draw the game
    def draw(self, snake_number, snake):
        snake = np.copy(snake[snake_number])
        apple = snake[0]
        head = snake[1]
        body = snake[2:]

        # draw everything
        background = np.zeros((s_size[0], s_size[1], 3), dtype=np.uint8)
        background[int(apple[0]-1), int(apple[1]-1)] = s_red
        background[int(head[0]-1), int(head[1]-1)] = s_green

        for i in range(len(body)):
            if np.any(body[i] == 0):
                break
            background[int(body[i, 0]-1), int(body[i, 1]-1)] = 80

        return background

    def screen(self, background):
        # show screen
        game = cv2.resize(background, (500, 500), interpolation=cv2.INTER_NEAREST)
        game = np.uint8(game)
        cv2.imshow("game", game)
        cv2.moveWindow("game", -520, 40)
        cv2.waitKey(s_wait_time)

        return

    # make an info graf
    def print_graf(self, avg, step, epsilon, step_limit, e):
        # save points
        self.avg_scores.append(avg)
        self.avg_step.append(step)
        self.test_count.append(e)

        # increase difficulty when
        score_limit = round(step_limit * 0.5)
        step_limit = round(step_limit * 0.8)

        # plt prints
        epsilon = round(epsilon, 3)
        plt.title(f'Epsilon {epsilon}', loc='right')
        plt.title(f'Min steps: {step_limit} Min scores: {score_limit}', loc='left')
        plt.xlabel(f'Games played {e*s_game_amount}')
        plt.ylabel("Scores / Steps")

        plt.grid(True)

        # plot scores
        plt.plot(self.test_count, self.avg_scores, label=f'Scores {round(avg,2)}')
        plt.plot(self.test_count, self.avg_step, label=f'Steps {round(step,2)}')

        # show
        plt.legend()
        plt.show(block=False)
        plt.pause(0.01)
        plt.cla()

        return


