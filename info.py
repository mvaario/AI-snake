import time

from settings import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


class info:
    def __init__(self, tf):
        self.avg_scores = []
        self.episodes = []
        self.avg_step = []

        self.tensorflow_setups(tf)

    def tensorflow_setups(self, tf):
        # gpus = tf.config.list_physical_devices('GPU')
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])

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
        if s_save_model:
            print(f'Model saverate: {s_save_rate} with name: {s_save_model_name}')
        else:
            print("Saving is OFF")
        if s_testing_ai:
            print(f'Test rate: {s_test_rate}')
        else:
            print("Testing is OFF")
        print("")
        time.sleep(1)
        return

    # draw the game
    def draw(self, snake_number, snake):
        snake = snake[snake_number]
        apple = snake[0]
        head = snake[1]
        body = snake[2:]

        # draw everything
        background = np.zeros((s_size[0], s_size[1], 3), dtype=np.uint8)
        background[int(apple[0]), int(apple[1])] = s_red
        background[int(head[0]), int(head[1])] = s_green

        for i in range(len(body)):
            if np.all(body[i] == -1):
                break
            else:
                background[int(body[i, 0]), int(body[i, 1])] = 80
        return background

    def screen(self, background):
        # show screen
        game = cv2.resize(background, (500, 500), interpolation=cv2.INTER_NEAREST)
        game = np.uint8(game)
        cv2.imshow("game", game)
        cv2.moveWindow("game", -520, 40)
        # cv2.moveWindow("game", 520, 40)
        cv2.waitKey(s_wait_time)

        return

    # make an info graf
    def print_graf(self, avg, step, epsilon, step_limit):
        # save points
        self.avg_scores.append(avg)
        self.episodes.append(len(self.avg_scores)-1)
        self.avg_step.append(step)

        x = 0.8 * step_limit

        # plt prints
        epsilon = round(epsilon, 3)
        plt.title(f'Epsilon {epsilon}', loc='right')
        plt.title(f'Step limit {step_limit} - {round(x, 2)}', loc='left')
        plt.xlabel(f'Episodes {len(self.episodes)}')
        plt.ylabel("Scores / Steps")

        plt.grid(True)

        # plot scores
        plt.plot(self.episodes, self.avg_scores, label=f'Scores {round(avg,2)}')
        plt.plot(self.episodes, self.avg_step, label=f'Steps {round(step,2)}')

        # show
        plt.legend()
        plt.show(block=False)
        plt.pause(0.01)
        plt.cla()

        return


