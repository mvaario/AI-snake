from settings import *
import numpy as np
import time

class game:
    def __init__(self):
        self.snake = np.zeros([s_game_amount, s_size[0] * s_size[1], 2])
        self.done = np.ones(s_game_amount, dtype=bool)
        self.point = np.zeros(s_game_amount, dtype=bool)

        self.last_position = np.zeros([s_game_amount, 2])
        self.distance_score = s_distance_score
        self.random_poit = s_random_point

        # all available spots
        self.all_spots = []
        for i in range(0, s_size[0]*s_size[1]):
            self.all_spots.append(i+1)

    # spawn head and body
    def spawn_snake(self, snake_number):
        # Clear old snake
        self.snake[snake_number, 0:] = 0
        self.done[snake_number] = False
        self.point[snake_number] = False

        # spawn snake head
        head = np.random.randint(1, [s_size[0]+1, s_size[1]+1], size=2)

        snake_body = np.array([-1, -1])
        while np.any(snake_body <= 0) or snake_body[0] > s_size[0] or snake_body[1] > s_size[1]:
            dir = np.random.randint(1, 5)
            if dir == 1:
                snake_body[0] = head[0] + 1
                snake_body[1] = head[1]
            elif dir == 2:
                snake_body[0] = head[0] - 1
                snake_body[1] = head[1]
            elif dir == 3:
                snake_body[0] = head[0]
                snake_body[1] = head[1] + 1
            elif dir == 4:
                snake_body[0] = head[0]
                snake_body[1] = head[1] - 1
            else:
                print("Error in spawn snake")
                print(dir)
                quit()
        if s_max_len > 0:
            self.snake[snake_number, 2] = snake_body
        self.snake[snake_number, 1] = head

        return

    # spawn apple
    def spawn_apple(self, snake_number):
        # get correct snake
        snake = self.snake[snake_number, 1:]

        # convert snake coordinates to work with numpy set diff
        snake_body = []
        for i in range(len(snake)):
            if np.all(snake[i] == 0):
                break
            y = snake[i, 0] - 1
            y = y * s_size[0]
            body = y + snake[i, 1]
            snake_body.append(body)

        snake_body = np.array(snake_body)

        # get point from all spots - snake
        available_spawns = np.setdiff1d(self.all_spots, snake_body)
        # if game is completed
        if len(available_spawns) == 0:
            print("you think you are pro beating shit game whit", s_state_size, "size game...")
            quit()

        # get random apple position from available spawns
        apple = np.random.choice(available_spawns, 1)
        apple = apple[0]

        # convert apple number to coordinates
        y = 1
        while apple > s_size[1]:
            y += 1
            apple -= s_size[1]

        apple = [y, apple]

        # set apple position is in front
        self.snake[snake_number, 0] = apple

        return

    # movements
    def move_snake(self, action, snake_number):
        snake = np.copy(self.snake[snake_number, 1:])
        snake_copy = np.copy(snake)

        # get movement direction
        if action == 0:
            snake[0, 0] += 1
        elif action == 1:
            snake[0, 0] -= 1
        elif action == 2:
            snake[0, 1] += 1
        elif action == 3:
            snake[0, 1] -= 1
        else:
            print("Error in action")
            print(action)
            quit()

        # move snake
        if np.all(snake[0] == snake[1]):
            done = True
        else:
            done = False
            # move snake body
            for i in range(len(snake)-1):
                if np.any(snake[i+1] == 0):
                    # save last position for adding snake
                    self.last_position[snake_number] = snake_copy[i]
                    break
                snake[i+1] = snake_copy[i]

        # move head
        self.snake[snake_number, 1:] = snake
        return done

    # check collision
    def check(self, snake_number, done):
        point = False
        if done:
            # mark if game is done
            self.done[snake_number] = True
            return point, done

        # point = got the apple / done = dead
        snake = self.snake[snake_number]
        done = self.done[snake_number]

        head = snake[1]
        # Check walls
        if np.any(head > s_size[0]):
            done = True
        elif np.any(head <= 0):
            done = True
        else:
            # Check if snake hit itself
            for i in range(len(snake)-2):
                if np.all(snake[1] == snake[i+2]):
                    done = True

        # check apple
        if not done:
            apple = snake[0]
            if np.all(head == apple):
                point = True
                # add snake
                if len(snake) < s_max_len:
                    for i in range(len(snake)):
                        if snake[i] == 0:
                            snake[i] = self.last_position[snake_number]
                            break
                # spawn new apple
                game.spawn_apple(self, snake_number)

        # save results
        self.point[snake_number] = point
        self.done[snake_number] = done
        return point, done

    # calculate rewards
    def reward_calculation(self, point, snake_number):
        done = self.done[snake_number]
        snake = self.snake[snake_number]

        step_reward = 0
        if done:
            step_reward += s_penalty
        elif point:
            step_reward += s_apple_score
        else:
            # Calculate distance to the apple
            apple = snake[0]
            head = snake[1]

            # previous head position
            if np.all(snake[2] == 0):
                last_head = self.last_position[snake_number]
            else:
                last_head = snake[2]

            score = self.distance_score

            distance = abs(apple - head)
            old_distance = abs(apple - last_head)

            difference = old_distance - distance
            step_reward += difference[0] * score
            step_reward += difference[1] * score
            if step_reward < 0:
                step_reward *= 2
            step_reward = int(step_reward)
            # add snake even without the apple (doesn't affect rewards)
            # if not done:
                # if np.random.rand() < self.random_poit:
                #     self.point[snake_number] = True

        return step_reward
