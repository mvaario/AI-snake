from settings import *
import numpy as np


class game:
    def __init__(self):
        self.snake = np.ones([s_game_amount, s_size[0] * s_size[1], 2])
        self.snake = np.negative(self.snake)
        self.done = np.ones([s_game_amount, 1], dtype=bool)
        self.point = np.zeros([s_game_amount, 1], dtype=bool)

        self.distance_score = s_distance_score
        self.random_poit = s_random_point

    # spawn head and body
    def spawn_snake(self, snake_number):
        # Clear old snake
        self.snake[snake_number, 0:] = -1
        self.done[snake_number] = False
        self.point[snake_number] = False

        # spawn snake head
        head = np.random.randint(1, s_size[1] - 1, size=2)

        # spawn snake body
        snake = np.array([-1, -2])
        while np.any(0 > snake) or np.any(snake > (s_size[0] - 1)):
            snake = np.random.randint(0, 4)
            if snake == 0:
                snake = np.array([head[0] - 1, head[1]])
            elif snake == 1:
                snake = np.array([head[0] + 1, head[1]])
            elif snake == 2:
                snake = np.array([head[0], head[1] - 1])
            else:
                snake = np.array([head[0], head[1] + 1])

        self.snake[snake_number, 1] = head
        self.snake[snake_number, 2] = snake
        return

    # spawn apple
    def spawn_apple(self, snake_number):
        # get correct snake
        snake = self.snake[snake_number]
        snake = snake[1:]

        # spawn apple
        apple = np.random.randint(0, s_size[0], size=2)

        # check apple position
        for i in range(len(snake)):
            if np.all(snake[i] == -1):
                break
            if np.all(apple == snake[i]):
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # let's change apple position and see if it works, (got looping without)
                for i in range(s_size[0]):
                    if apple[1] == s_size[0]-1:
                        break
                    apple[1] += 1
                    if np.all(apple != snake[i]):
                        break
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # Spawn new apple
                if np.all(apple == snake[i]):
                    game.spawn_apple(self, snake_number)

        # set apple position is in front
        self.snake[snake_number, 0] = apple
        return

    # movements
    def move_snake(self, action, snake_number):
        # Again no idea why it needs to be copied...
        snake = np.copy(self.snake[snake_number])

        # get movement direction (action size = 4)
        if action == 0:
            movement = [-1, 0]
        elif action == 1:
            movement = [1, 0]
        elif action == 2:
            movement = [0, -1]
        elif action == 3:
            movement = [0, 1]
        else:
            print("Wrong action")
            quit()

        # # # # # # # # # # # # # # # # # # # # # # # #
        # get movement direction (action size = 3)
        # movement = [0, 0]
        # if action == 1:
        #     movement = direction
        # else:
        #     movement[0] = direction[1]
        #     movement[1] = direction[0]
        #     if action == 0:
        #         if direction[0] == 0:
        #             movement[0] *= -1
        #             movement[1] *= -1
        #     if action == 2:
        #         if direction[1] == 0:
        #             movement[0] *= -1
        #             movement[1] *= -1
        #
        # movement[0] = int(movement[0])
        # movement[1] = int(movement[1])
        # # # # # # # # # # # # # # # # # # # # # # # #

        # movement backwards -> dead
        back = snake[1] + movement
        if np.all(back == snake[2]):
            done = True
        else:
            done = False
            # move snake body
            snake_copy = np.copy(snake)
            for i in range(len(snake)):
                if np.all(snake[i] == -1) and i > 0:
                    if self.point[snake_number]:
                        snake[i] = last_position
                    break
                elif i > 1:
                    x = i - 1
                    snake[i] = snake_copy[x]
                    last_position = snake_copy[i]

            # move head
            snake[1] = snake[1] + movement
            self.snake[snake_number] = snake

        return done

    # check collision
    def check(self, snake_number, done):
        point = False
        if done:
            # mark if game is done
            self.done[snake_number] = done
            return point, done

        # point = got the apple / done = dead
        snake = self.snake[snake_number]
        done = self.done[snake_number]

        head = snake[1]

        # Check walls
        if head[0] == s_size[0]:
            done = True
        elif head[1] == s_size[1]:
            done = True
        elif head[0] < 0:
            done = True
        elif head[1] < 0:
            done = True
        else:
            # Check if snake hit itself
            for i in range(len(snake) - 2):
                if np.all(snake[i] == -1):
                    break
                k = i + 2
                if np.all(head == snake[k]):
                    done = True

        # check apple
        if not done:
            apple = snake[0]
            if np.all(head == apple):
                point = True
                # spawn new apple
                game.spawn_apple(self, snake_number)

        # save results
        self.point[snake_number] = point
        self.done[snake_number] = done
        # add snake even without the apple (doesn't affect rewards)
        if not done:
            if np.random.rand() < self.random_poit:
                self.point[snake_number] = True

        return point, done

    # calculate rewards
    def reward_calculation(self, point, snake_number):
        done = self.done[snake_number]
        snake = self.snake[snake_number]

        step_reward = 0
        if done:
            step_reward -= s_penalty
        elif point:
            step_reward += s_apple_score
        else:
            # Calculate distance to the apple
            apple = snake[0]
            head = snake[1]
            last_head = snake[2]
            score = self.distance_score

            distance = abs(apple - head)
            old_distance = abs(apple - last_head)
            difference = old_distance - distance
            step_reward += difference[0] * score
            step_reward += difference[1] * score
            # to avoid wander with zero loss
            if step_reward < 0:
                step_reward *= 2.2

        return step_reward
