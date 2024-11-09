import cv2
from game import *
from DQNAgent import *
from info import *
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import time
from keyboard import *
from PPOModel import *


class MAIN:
    def __init__(self):
        # define state
        self.ori_state = np.zeros(s_state_size)
        # rewards and step
        self.step = np.zeros([s_game_amount, 1])
        self.step_limit = s_start_step_limit

        self.validation_reward = 0

        # sort order
        self.sort_order = []

    # Create state using the closest snake positions
    def create_custom_state(self, snake, done):
        if done:
            state = np.zeros(s_state_size)
            return state

        state = []
        lengths = []
        head_coordinates = snake[1]
        for i in range(len(snake)):
            # state done
            if np.all(snake[i] == 0):
                break

            # add snake point to the state
            state.append(snake[i, 0])
            state.append(snake[i, 1])

            # get distance from head
            body_coordinates = snake[i]
            length = abs(body_coordinates - head_coordinates)
            lengths.append(sum(length))

        state = np.array(state)

        # sort snake body
        if s_sort_state:
            state = np.reshape(state, (-1, 2))
            snake_body = state[2:]
            state = state[:2]

            # if new:
            lengths = lengths[2:]
            lengths = np.array(lengths)
            self.sort_order = np.argsort(lengths)

            snake_body = snake_body[self.sort_order]
            state = np.concatenate((state, snake_body), axis=0)

        state = np.reshape(state, (-1))

        # check state size
        if len(state) > s_state_size:
            state = state[:s_state_size]
        elif len(state) < s_state_size:
            k = 4
            while len(state) < s_state_size:
                if len(state) > 4:
                    state = np.append(state, state[k])
                    k += 1
                else:
                    state = np.append(state, 0)

        # normalize values
        state = np.array(state) / s_size[1]
        state = np.reshape(state, s_state_size)
        print(state)
        input()

        return state

    # Create image like state
    def create_state(self, snake, apple, done):
        # this is that one bs thing... without using copy snake will change
        snake_copy = np.copy(snake)
        apple_copy = np.copy(apple)
        state = np.zeros(s_state_size)
        if done:
            return state
        # (y, x)
        # 1 - 20
        for i in range(len(snake_copy)):
            position = snake_copy[i]
            if sum(position) == 0:
                break
            position[0] -= 1
            position[1] -= 1
            if i == 0:
                state[int(position[0]), int(position[1])] = 2
            else:
                state[int(position[0]), int(position[1])] = 1

        for i in range(len(apple_copy)):
            position = apple_copy[i]
            if sum(position) == 0:
                break
            position[0] -= 1
            position[1] -= 1
            state[int(position[0]), int(position[1])] = 3

        return state

    # game states
    def game_states(self, snake_number, r_testing):
        # create state
        state = main.create_state(game.snake[snake_number], game.apple[snake_number], game.done[snake_number])

        # pick action
        if s_control and snake_number == 0 and r_testing:
            background = info.draw(snake_number, game.snake, game.apple)
            action = None
            print("keyboard control enabled")
            while action == None:
                info.screen(background)
                action = keyboard.get_keys()
        else:
            if s_use_ppo:
                action, value, log_prob = PPO.get_action(state, r_testing)
            else:
                action = DQNA.get_qs(state, r_testing)

        # print(game.snake[snake_number])
        # move snake
        game.move_snake(snake_number, action)

        # check snake
        point = game.check(snake_number)

        # add snake
        game.add_snake(snake_number, point)

        # reward calculations
        step_reward = game.reward_calculation(snake_number, point)

        if not r_testing:
            # create new state
            next_state = main.create_state(game.snake[snake_number], game.apple[snake_number], game.done[snake_number])

            # update memory
            if s_use_ppo:
                PPO.update_memory(state,
                                  action,
                                  log_prob,
                                  value,
                                  step_reward,
                                  game.done[snake_number]
                                  )
            else:
                DQNA.update_replay_memory(state,
                                          action,
                                          step_reward,
                                          next_state,
                                          game.done[snake_number]
                                          )

        else:
            self.validation_reward += point

        # reset steps and step limit
        if game.done[snake_number]:
            self.step[snake_number] = 0
        else:
            self.step[snake_number] += 1
            if self.step[snake_number] >= self.step_limit:
                game.done[snake_number] = True
                self.step[snake_number] = 0

        return

    # reset all for testing
    def reset(self, games):
        self.step = np.zeros([games, 1])
        self.validation_reward = 0

        game.snake = np.zeros([games, s_max_len, 2])
        game.apple = np.zeros([games, s_apple_amount, 2])
        game.done = np.ones(games, dtype=bool)

        game.last_position = np.zeros([games, 2])
        return

    # testing the AI with new games
    def model_validation(self):
        # create games
        for snake_number in range(s_test_games):
            game.spawn_snake(snake_number)
            game.spawn_apple(snake_number)

        # play all the games one time
        steps = 0
        while not np.all(game.done):
            # one step every game
            for snake_number in range(s_test_games):
                if not game.done[snake_number]:
                    # show game
                    if snake_number == 0:
                        background = info.draw(snake_number, game.snake, game.apple)
                        info.screen(background)
                    main.game_states(snake_number, r_testing=True)
                    steps += 1
        return steps

    # increase difficulty based on test results
    def increase_difficulty(self, steps, e):
        avg = main.validation_reward / s_test_games
        step = steps / s_test_games

        # check limits
        step_limit = self.step_limit * s_step_limit_multiplier
        # score calculation
        score_limit = (step_limit / s_size[0]) * s_score_limit_multiplier

        # if steps and scores are good enough increase difficulty
        if step >= step_limit and avg >= score_limit and e != 1:
            self.step_limit += s_step_increase

            # calculate new step limit
            step_limit = self.step_limit * s_step_limit_multiplier

            # calculate new score limit
            score_limit = (step_limit / s_size[0]) * s_score_limit_multiplier

            # increase snake start length
            game.add_len += s_add_len
            if game.add_len > 75:
                # if more than 75 possibly to die on start
                game.add_len = 75

            # increase exploration
            if not s_use_ppo:
                DQNA.epsilon += s_epsilon_increase
                if DQNA.epsilon > 1:
                    DQNA.epsilon = 1

        # save test results for graf
        info.step_limit = step_limit
        info.score_limit = score_limit
        info.avg_scores.append(avg)
        info.avg_step.append(step)
        info.episodes.append(e)

        return


if __name__ == '__main__':
    # initialize
    main = MAIN()
    info = INFO(tf)
    game = GAME()
    keyboard = KEYBOARD()
    if s_use_ppo:
        PPO = PPOModel()
    else:
        DQNA = DQNAgent()

    start = time.time()
    # define how many episodes (not games, all games with one step)
    for e in tqdm(range(1, s_episodes + 1), ascii=True, unit=' steps'):
        # make multiple games at once
        for snake_number in range(s_game_amount):
            # create a new game if needed
            if game.done[snake_number]:
                game.spawn_snake(snake_number)
                game.spawn_apple(snake_number)

            # all game states
            main.game_states(snake_number, r_testing=False)

        # Train model
        if s_use_ppo:
            current_lr = PPO.train_model(e)
            if e % s_test_rate == 0 or e == 1:
                info.ppo_lr_rate = float(current_lr)
        else:
            DQNA.train_model(e)

        # test the model
        if s_test_model:
            if e % s_test_rate == 0 or e == 1:
                # reset values for testing
                main.reset(games=s_test_games)

                # model validation
                steps = main.model_validation()

                # increase difficulty
                main.increase_difficulty(steps, e)

                # reset values for training
                main.reset(games=s_game_amount)

                # check graf lengths FIX THESE
                info.balance_episodes()
                info.balance_steps()
                info.balance_scores()

                # draw graf
                if not s_use_ppo:
                    info.print_graf(DQNA.epsilon, e, game.add_len)
                else:
                    info.ppo_graf(e, game.add_len)

