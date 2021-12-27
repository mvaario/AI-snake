import cv2
import numpy as np
from settings import *

class game:
    def __init__(self):
        self.head = []
        self.snake = []
        self.apple = []
        self.snake_add = []

        self.snake_len = 0

        self.old_snake = np.array([0, 0])

    # spawn head and body
    def spawn_snake(self):
        # spawn snake head
        self.head = np.random.randint(1, size[1] - 1, size=2)

        # spawn snake body
        snake = np.array([-1, -2])
        while np.any(1 > snake) or np.any(snake > (size[0] - 2)):
            snake = np.random.randint(0, 4)
            if snake == 0:
                snake = np.array([self.head[0] - 1, self.head[1]])
            elif snake == 1:
                snake = np.array([self.head[0] + 1, self.head[1]])
            elif snake == 2:
                snake = np.array([self.head[0], self.head[1] - 1])
            else:
                snake = np.array([self.head[0], self.head[1] + 1])

        self.snake = snake
        return

    # spawn apple
    def spawn_apple(self):
        # spawn apple
        apple = np.random.randint(1, size[0] - 2, size=2)
        self.snake_len = int(len(self.snake) / 2)

        # check apple position
        for i in range(self.snake_len):
            x = i * 2
            if apple[0] == self.snake[x] and apple[1] == self.snake[x + 1]:
                apple = self.head

        if np.all(apple == self.head):
            game.spawn_apple(self)

        self.apple = apple

        return

    # movements
    def move_snake(self, action, direction):
        movement = [0, 0]
        if action == 1:
            movement = direction
        else:
            movement[0] = direction[1]
            movement[1] = direction[0]
            if action == 0:
                if direction[0] == 0:
                    movement[0] *= -1
                    movement[1] *= -1
            if action == 2:
                if direction[1] == 0:
                    movement[0] *= -1
                    movement[1] *= -1

        # save last snake
        self.snake_add = np.array([self.snake[len(self.snake) - 2], self.snake[len(self.snake) - 1]])

        # copy snake
        self.old_snake = np.array(self.snake)
        # move first part of the snake
        self.snake[0] = self.head[0]
        self.snake[1] = self.head[1]

        # save snake length
        self.snake_len = int(len(self.old_snake) / 2)
        snake_len = self.snake_len - 1

        # move rest of the snake
        if snake_len > 0:
            for i in range(snake_len):
                x = i * 2
                self.snake[x + 2] = self.old_snake[x + 0]
                self.snake[x + 3] = self.old_snake[x + 1]

        # move head
        self.head = self.head + movement

        return

    # check collision
    def check(self):
        # point = got the apple / done = dead
        point = False
        done = False

        # check walls
        if self.head[0] == size[0]-1:
            done = True
        elif self.head[1] == size[1]-1:
            done = True
        elif self.head[0] <= 0:
            done = True
        elif self.head[1] <= 0:
            done = True
        else:
            # check if snake hit it self
            for i in range(self.snake_len):
                x = i * 2
                if self.head[0] == self.snake[x] and self.head[1] == self.snake[x + 1]:
                    done = True

        if not done:
            # check apple
            if np.all(self.head == self.apple):
                point = True
                # add to snake
                self.snake = np.append(self.snake, self.snake_add)
                # spawn new apple
                game.spawn_apple(self)

        return done, point

    # draw the game
    def draw(self):
        # draw everything
        background = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        background[self.apple[0], self.apple[1]] = red
        background[self.head[0], self.head[1]] = green
        for i in range(self.snake_len):
            x = i * 2
            background[self.snake[x], self.snake[x + 1]] = 80
        return background

