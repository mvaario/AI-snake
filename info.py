# import time
# import keras.backend
from settings import *
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from typing import List


class INFO:
    def __init__(self, tf):
        self.score_limit = 0
        self.step_limit = 0
        self.avg_scores = []
        self.avg_step = []
        self.last_score = 0
        self.last_step = 0
        self.episodes = []
        self.tensorflow_setups(tf)

        self.all_episodes = []
        self.episodes_1 = []
        self.episodes_2 = []
        self.episodes_3 = []
        self.episodes_4 = []

        self.all_scores = []
        self.scores_1 = []
        self.scores_2 = []
        self.scores_3 = []
        self.scores_4 = []

        self.all_steps = []
        self.step_1 = []
        self.step_2 = []
        self.step_3 = []
        self.step_4 = []


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
        if s_test_model:
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
        background[int(apple[0] - 1), int(apple[1] - 1)] = s_red
        background[int(head[0] - 1), int(head[1] - 1)] = s_green

        for i in range(len(body)):
            if np.any(body[i] == 0):
                break
            background[int(body[i, 0] - 1), int(body[i, 1] - 1)] = 80

        return background

    def screen(self, background):
        # show screen
        game = cv2.resize(background, (500, 500), interpolation=cv2.INTER_NEAREST)
        game = np.uint8(game)
        cv2.imshow("game", game)
        # cv2.moveWindow("game", -520, 40)
        cv2.waitKey(s_wait_time)

        return

    def smooth(scalars: List[float], weight: float) -> List[float]:
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    # make an info graf
    def print_graf(self, epsilon, e, add_len):
        # limits for difficult increase
        score_limit = round(self.score_limit)
        step_limit = round(self.step_limit)
        add_len = round(add_len)

        # plt prints
        epsilon = round(epsilon, 3)
        plt.title(f'Epsilon {epsilon}', loc='right')
        plt.title(f'Min steps: {step_limit} scores: {score_limit} len: {add_len}', loc='left')
        plt.xlabel(f'Episodes {e}   games: {s_game_amount}')
        plt.ylabel("Scores / Steps")

        plt.grid(True)

        # Smooth the data
        # smoothed_scores = info.smooth(self.all_scores, 0.5)
        # smoothed_steps = info.smooth(self.all_steps, 0.5)

        # plot scores
        plt.plot(self.all_episodes, self.all_scores, label=f'Scores {round(self.last_score, 2)}')
        plt.plot(self.all_episodes, self.all_steps, label=f'Steps {round(self.last_step, 2)}')

        # show
        plt.legend()
        plt.show(block=False)

        plt.pause(0.01)
        plt.cla()

        return

    # balance steps fix this
    def balance_steps(self):
        step = self.avg_step
        step_1 = self.step_1
        step_2 = self.step_2
        step_3 = self.step_3
        step_4 = self.step_4

        e, step_4 = self.calculation(step_4, count=200)
        step_4 = np.append(step_4, e)

        e, step_3 = self.calculation(step_3, count=100)
        step_4 = np.append(step_4, e)

        e, step_2 = self.calculation(step_2, count=50)
        step_3 = np.append(step_3, e)

        e, step_1 = self.calculation(step_1, count=20)
        step_2 = np.append(step_2, e)

        e, step = self.calculation(step, count=10)
        step_1 = np.append(step_1, e)

        self.avg_step = step
        self.step_1 = step_1
        self.step_2 = step_2
        self.step_3 = step_3
        self.step_4 = step_4

        self.all_steps = np.concatenate([step_4, step_3, step_2, step_1, step])

    # balance scores fix this
    def balance_scores(self):
        scores = self.avg_scores
        scores_1 = self.scores_1
        scores_2 = self.scores_2
        scores_3 = self.scores_3
        scores_4 = self.scores_4

        e, scores_4 = self.calculation(scores_4, count=200)
        scores_4 = np.append(scores_4, e)

        e, scores_3 = self.calculation(scores_3, count=100)
        scores_4 = np.append(scores_4, e)

        e, scores_2 = self.calculation(scores_2, count=50)
        scores_3 = np.append(scores_3, e)

        e, scores_1 = self.calculation(scores_1, count=20)
        scores_2 = np.append(scores_2, e)

        e, scores = self.calculation(scores, count=10)
        scores_1 = np.append(scores_1, e)

        self.avg_scores = scores
        self.scores_1 = scores_1
        self.scores_2 = scores_2
        self.scores_3 = scores_3
        self.scores_4 = scores_4

        self.all_scores = np.concatenate([scores_4, scores_3, scores_2, scores_1, scores])

    # balance episodes fix this
    def balance_episodes(self):
        episodes = self.episodes
        episodes_1 = self.episodes_1
        episodes_2 = self.episodes_2
        episodes_3 = self.episodes_3
        episodes_4 = self.episodes_4

        e, episodes_4 = self.calculation(episodes_4, count=200)
        episodes_4 = np.append(episodes_4, e)

        e, episodes_3 = self.calculation(episodes_3, count=100)
        episodes_4 = np.append(episodes_4, e)

        e, episodes_2 = self.calculation(episodes_2, count=50)
        episodes_3 = np.append(episodes_3, e)

        e, episodes_1 = self.calculation(episodes_1, count=20)
        episodes_2 = np.append(episodes_2, e)

        e, episodes = self.calculation(episodes, count=10)
        episodes_1 = np.append(episodes_1, e)

        self.episodes = episodes
        self.episodes_1 = episodes_1
        self.episodes_2 = episodes_2
        self.episodes_3 = episodes_3
        self.episodes_4 = episodes_4

        self.all_episodes = np.concatenate([episodes_4, episodes_3, episodes_2, episodes_1, episodes])
        return

    def calculation(self, episodes, count):
        e = []
        if len(episodes) >= count:
            half = int(len(episodes) / 2)
            for i in range(half):
                k = i * 2
                e_avg = (episodes[k] + episodes[k + 1]) / 2
                e.append(e_avg)
            episodes = []

        return e, episodes

    # def balance_episodes(self, num_episodes=100, count=10):
    #     print(self.episodes)
    #     for _ in range(num_episodes):
    #         for i in range(4, -1, -1):  # Iterate over all episodes from index 4 to 0
    #             e, self.episodes[i] = self.calculation(self.episodes[i], count)
    #             if i > 0:  # Only extend if not the last episode
    #                 self.episodes[i - 1].extend(e)
    #             else:  # If it's the last episode, extend all_episodes
    #                 self.episodes[i].extend(e)
    #         # Extend all_episodes after each iteration of the outer loop
    #         self.all_episodes.extend([])
    #
    #         print(self.episodes)
    #
    #     quit()
    #
    # def calculation(self, episodes, count):
    #     e = []
    #     if len(episodes) >= count:
    #         half = len(episodes) // 2
    #         for i in range(half):
    #             e_avg = (episodes[i * 2] + episodes[i * 2 + 1]) / 2
    #             e.append(e_avg)
    #         episodes = e[:]  # Update episodes with the new episodes
    #     print(e, episodes)
    #     quit()
    #     return e, episodes
























