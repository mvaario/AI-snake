from settings import *
import numpy as np
import cv2
import matplotlib.pyplot as plt


class info:
    def __init__(self):
        self.scores = []
        self.avg = []
        self.episodes = []
        self.step = []

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
        cv2.waitKey(s_wait_time)

        return

    # make an info graf
    def print_graf(self, steps, epsilon):
        # calculate test games average
        avg = np.average(self.scores)
        avg = avg / s_test_games
        self.avg.append(avg)
        self.scores = []

        self.episodes.append(len(self.avg))
        step = steps / s_test_games
        self.step.append(step)

        # plt prints
        epsilon = round(epsilon, 3)
        plt.title(f'Epsilon {epsilon}', loc='right')
        plt.xlabel(f'Episodes {len(self.episodes)}')
        plt.ylabel("Scores / Steps")

        plt.grid(True)

        # plot scores
        plt.plot(self.episodes, self.avg, label='Scores')
        plt.plot(self.episodes, self.step, label='Steps')

        # show
        plt.legend()
        plt.show(block=False)
        plt.pause(0.01)
        plt.cla()

        return


