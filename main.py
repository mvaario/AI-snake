import cv2
import threading
from game import *
from DQNAgent import *
from info import *
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import time


class main:
    def __init__(self):
        # define state
        self.ori_state = np.zeros(s_state_size)
        # rewards and step
        self.step = np.zeros([s_game_amount, 1])
        self.ep_reward = 0
        self.step_limit = s_step_limit

    # Create state
    def create_state(self, snake):


        state = []
        lengths = []
        head_coordinates = np.sum(snake[0])
        for i in range(len(snake)):
            if len(state) < s_state_size:
                state.append(snake[i, 0])
                state.append(snake[i, 1])

                # get distance from head
                body_coordinates = np.sum(snake[i])
                lengths.append(abs(body_coordinates - head_coordinates))
            else:
                # if state is full and get 0
                if np.all(snake[i] == 0):
                    break
                # check if closer than max
                max = np.max(lengths)
                body_coordinates = np.sum(snake[i])
                current_len = abs(body_coordinates - head_coordinates)
                # change max
                if max > current_len:
                    for k in range(len(lengths)):
                        if lengths[k] == max:
                            lengths[k] = current_len
                            state[k] = snake[i]

        state = np.array(state)
        state = state / s_size[1]

        # if snake out of map
        if np.any(state > 1) or np.any(state < 0):
            for i in range(len(state)):
                if state[i] > 1:
                    state[i] = 2
                elif state[i] < 0:
                    state[i] = 0

        return state

    # game states
    def game_states(self, snake_number, r_testing):
        # create state
        state = main.create_state(game.snake[snake_number])

        # pick action
        action = DQNA.get_qs(state, r_testing)

        # if snake_number == 0:
        #     background = info.draw(snake_number, game.snake)
        #     info.screen(background)
        #     action = int(input())

        # move snake
        done = game.move_snake(action, snake_number)

        # check snake
        point, done = game.check(snake_number, done)

        # reward calculations
        step_reward = game.reward_calculation(point, snake_number)

        # add step wards to episode reward
        self.ep_reward += step_reward

        if not r_testing:
            # create new state
            next_state = main.create_state(game.snake[snake_number])

            # update memory
            DQNA.update_replay_memory(state,
                                      action,
                                      step_reward,
                                      next_state,
                                      done
                                      )

        # reset steps and step limit
        if game.done[snake_number]:
            self.step[snake_number] = 0
        else:
            main.step[snake_number] += 1
            if main.step[snake_number] >= main.step_limit:
                game.done[snake_number] = True
                self.step[snake_number] = 0

        return

    # reset all for testing
    def reset(self, games):
        # testing uses different number of games
        self.step = np.zeros([games, 1])
        self.ep_reward = 0

        game.snake = np.zeros([games, s_size[0] * s_size[1], 2])
        game.done = np.ones(games, dtype=bool)
        game.point = np.zeros(games, dtype=bool)

        game.last_position = np.zeros([games, 2])
        return

    # testing the AI with new games
    def testing_ai(self, e):
        r_testing = True
        steps = 0

        # reset values to test mode
        main.reset(s_test_games)
        for snake_number in range(s_test_games):
            game.spawn_snake(snake_number)
            game.spawn_apple(snake_number)

        # play all the games one time
        while not np.all(game.done):
            # one step every game
            for snake_number in range(s_test_games):
                if not game.done[snake_number]:
                    # show game
                    if snake_number == 0:
                        background = info.draw(snake_number, game.snake)
                        info.screen(background)
                        cv2.waitKey(s_wait_time)

                    main.game_states(snake_number, r_testing)
                    steps += 1

        # all games done
        cv2.destroyAllWindows()
        # increase difficulty
        avg, step = main.increase_difficulty(steps)

        # show graf
        info.print_graf(avg, step, DQNA.epsilon, main.step_limit, e)
        # reset values to train mode
        main.reset(s_game_amount)

        return

    def increase_difficulty(self, steps):
        avg = main.ep_reward / s_test_games
        step = steps / s_test_games

        limit = main.step_limit * 0.8
        score_limit = main.step_limit * 0.5
        # if steps and scores are good enough increase difficulty
        if step >= limit and avg >= score_limit:
            main.step_limit += 10

            # increase random point
            # game.random_poit += 0.05
            # if game.random_poit >= 0.75:
            #     game.random_poit = 0.75

            # decay distance score
            game.distance_score -= 0.05
            if game.distance_score <= 0:
                game.distance_score = 0

            # increase exploration
            DQNA.epsilon += 0.10
            if DQNA.epsilon > 1:
                DQNA.epsilon = 1

        return avg, step


if __name__ == '__main__':
    # initialize
    info = info(tf)
    main = main()
    game = game()
    input_shape = np.zeros(s_state_size)
    DQNA = DQNAgent(input_shape)

    # define how many episodes
    for e in tqdm(range(1, s_episodes + 1), ascii=True, unit='episodes'):
        r_testing = False
        games_done = 0
        # while all the games haven't ended
        while games_done < s_game_amount:
            # make multiple games at once
            for snake_number in range(s_game_amount):
                # create new game
                if game.done[snake_number]:
                    game.spawn_snake(snake_number)
                    game.spawn_apple(snake_number)

                main.game_states(snake_number, r_testing)

            # count when all the games have ended
            games_done += np.count_nonzero(game.done)

            # Train model
            DQNA.train_model()

        # episode end stuff
        main.ep_reward = 0

        # model modifications
        DQNA.epsilon_decay()
        DQNA.target_update(e)
        if s_save_model and e % s_save_rate == 0:
            # save model
            time.sleep(0.1)
            DQNA.model.save(f'D:\Programs\Coding\Projects\AI_snake\models\{s_save_model_name}_episode_{e:}.model')

        # # test the model
        if s_testing_ai:
            if e % s_test_rate == 0 or e <= 1:
                time.sleep(0.1)
                main.testing_ai(e)
