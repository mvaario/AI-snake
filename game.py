from settings import *
import numpy as np
import time


class GAME:
    def __init__(self):
        self.snake = np.zeros([s_game_amount, s_max_len, 2])
        self.apple = np.zeros([s_game_amount, s_apple_amount, 2])

        self.done = np.ones(s_game_amount, dtype=bool)

        self.last_position = np.zeros([s_game_amount, 2])
        self.add_len = s_start_len

        # need to be here or allow grow does not work with start len
        self.allow_snake_grow = s_allow_snake_grow

        # all available spots
        self.all_spots = []
        for i in range(0, s_size[0] * s_size[1]):
            self.all_spots.append(i + 1)

    # spawn head and body
    def spawn_snake(self, snake_number):
        # check parameters
        if s_max_len < self.add_len:
            print("Start len is bigger than max length")
            self.add_len = s_max_len

        # Clear old snake
        self.snake[snake_number, 0:] = 0
        self.done[snake_number] = False

        # spawn snake head
        head = np.random.randint(1, [s_size[0] + 1, s_size[1] + 1], size=2)
        self.snake[snake_number, 0] = head

        # save head as snake last position
        self.last_position[snake_number] = head

        # get random snake direction
        direction = np.random.randint(1, 5)
        last_direction = direction

        # loop catcher
        start_time = time.time()
        # from random direction get new position
        new_position = np.array([0, 0])
        while np.count_nonzero(self.snake[snake_number]) < (self.add_len + 1) * 2:
            if time.time() - start_time > 1:
                print("took too long to spawn snake, with len:", self.add_len)
                quit()
            if direction == 1:
                new_position[0] = self.last_position[snake_number, 0] + 1
                new_position[1] = self.last_position[snake_number, 1]
            elif direction == 2:
                new_position[0] = self.last_position[snake_number, 0] - 1
                new_position[1] = self.last_position[snake_number, 1]
            elif direction == 3:
                new_position[0] = self.last_position[snake_number, 0]
                new_position[1] = self.last_position[snake_number, 1] + 1
            elif direction == 4:
                new_position[0] = self.last_position[snake_number, 0]
                new_position[1] = self.last_position[snake_number, 1] - 1
            else:
                print("bad direction", direction)
                quit()

            add = True
            # check new position is on the grid
            if np.any(new_position <= 0) or new_position[0] > s_size[0] or new_position[1] > s_size[1]:
                add = False
                new_position = np.array([0, 0])
            else:
                # check new position is not taken
                for i in range(len(self.snake[snake_number])):
                    if np.all(self.snake[snake_number, i] == 0):
                        # this is only to save time
                        break
                    if np.all(self.snake[snake_number, i] == new_position):
                        add = False
                        new_position = np.array([0, 0])
                        break

            # add new position
            if add:
                # add new position
                self.last_position[snake_number] = new_position
                self.allow_snake_grow = True  # make sure we can add snake
                self.add_snake(snake_number, point=True)
                self.allow_snake_grow = s_allow_snake_grow  # change adding to back
                last_direction = direction
            else:
                # change direction
                if last_direction == 1 or last_direction == 2:
                    direction = np.random.randint(3, 5)
                else:
                    direction = np.random.randint(1, 3)

        return

    def available_positions(self, position_1, position_2):
        # convert position coordinates to numeric
        available_positions = position_2
        converted_positions = []
        for i in range(len(position_1)):
            if np.all(position_1[i] == 0):
                # just save time
                break
            y = position_1[i, 0] - 1
            y = y * s_size[0]
            converted = y + position_1[i, 1]
            converted_positions.append(converted)

        converted_positions = np.array(converted_positions)

        # get point from all positions
        available_positions = np.setdiff1d(available_positions, converted_positions)

        return available_positions

    # spawn apple
    def spawn_apple(self, snake_number):
        # remove snake positions from all positions
        snake = self.snake[snake_number]
        available_spawns = self.available_positions(snake, self.all_spots)

        # spawn multiple apples
        for i in range(s_apple_amount):
            apple = self.apple[snake_number]

            # check if apple needs to be spawned
            if np.all(apple[i] == 0):
                # remove apple positions from all positions
                available_spawns = self.available_positions(apple, available_spawns)

                # if game is completed - this could fail now since added multiple apples
                if len(available_spawns) == 0:
                    print("you think you are pro beating shit game whit", s_size, "size game...")
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

                # save apple positions
                self.apple[snake_number, i] = apple

        return

    # movements
    def move_snake(self, snake_number, action):
        snake = np.copy(self.snake[snake_number])
        snake_copy = np.copy(snake)
        self.last_position[snake_number] = snake[0]

        # move snake head
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
        if len(snake) > 1:
            if np.all(snake[0] == snake[1]):
                self.done[snake_number] = True
            else:
                # move snake body
                for i in range(len(snake) - 1):
                    # save last position for adding snake
                    if np.any(snake[i + 1] == 0):
                        self.last_position[snake_number] = snake_copy[i]
                        break
                    snake[i + 1] = snake_copy[i]

        self.snake[snake_number] = snake
        return

    # check collision
    def check(self, snake_number):
        done = self.done[snake_number]
        point = False
        if done:
            return point

        # point = got the apple / done = dead
        snake = self.snake[snake_number]
        apple = self.apple[snake_number]

        head = snake[0]
        body = snake[1:]

        # Check walls
        if np.any(head > s_size[0]):
            done = True
        elif np.any(head <= 0):
            done = True
        else:
            # Check if snake hit itself
            for i in range(len(body)):
                if np.all(head == body[i]):
                    done = True
                    break
                if np.all(body[i] == 0):
                    break

        # check apple
        if not done:
            for i in range(len(apple)):
                if np.all(apple[i] == 0):
                    break
                if np.all(head == apple[i]):
                    point = True
                    apple[i] = [0, 0]
                    self.spawn_apple(snake_number)
                    break

        # save done
        self.done[snake_number] = done
        return point

    # add snake
    def add_snake(self, snake_number, point):
        if not self.allow_snake_grow:
            return

        snake = self.snake[snake_number]
        body = snake[1:]
        # return if snake len is full or disabled snake grow
        if np.count_nonzero(body) > s_max_len * 2:
            return

        # point
        if point or np.count_nonzero(body) < (self.add_len * 2):
            # Add last position to snake body
            for i in range(s_max_len-1):
                if np.all(body[i] == 0):
                    body[i] = self.last_position[snake_number]
                    break
        return

    # calculate rewards
    def reward_calculation(self, snake_number, point):
        done = self.done[snake_number]
        snake = self.snake[snake_number]

        step_reward = 0
        if done:
            step_reward += s_penalty
        elif point:
            step_reward += s_apple_score
        # calculate distance reward
        else:
            # survive reward
            snake_len = np.count_nonzero(snake)
            snake_len -= 2
            snake_len /= 2
            if snake_len >= 1:
                step_reward += snake_len * s_length_score_multiplier

        return step_reward
