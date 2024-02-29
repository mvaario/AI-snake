from game import *
from DQNAgent import *
from info import *
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import time


class MAIN:
    def __init__(self):
        # define state
        self.ori_state = np.zeros(s_state_size)
        # rewards and step
        self.step = np.zeros([s_game_amount, 1])
        self.step_limit = s_start_step_limit

        self.validation_reward = 0

    # Create state
    def create_state(self, snake, done):
        if done:
            state = np.zeros(s_state_size)
            return state

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
                            k = k * 2
                            state[k] = snake[i, 0]
                            state[k + 1] = snake[i, 1]
                            break

        state = np.array(state)
        state = state / s_size[1]

        return state

    # game states
    def game_states(self, snake_number, r_testing):
        # create state
        state = main.create_state(game.snake[snake_number], game.done[snake_number])

        # pick action
        action = DQNA.get_qs(state, r_testing)

        # move snake
        game.move_snake(snake_number, action)

        # check snake
        point = game.check(snake_number)

        # add snake
        game.add_snake(snake_number, point)

        # reward calculations
        step_reward = game.reward_calculation(snake_number, point)

        if r_testing:
            self.validation_reward += step_reward
        else:
            # create new state
            next_state = main.create_state(game.snake[snake_number], game.done[snake_number])

            # update memory
            DQNA.update_replay_memory(state,
                                      action,
                                      step_reward,
                                      next_state,
                                      game.done[snake_number]
                                      )

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
    def reset(self):
        self.step = np.zeros([s_game_amount, 1])
        self.validation_reward = 0

        game.snake = np.zeros([s_game_amount, s_size[0] * s_size[1], 2])
        game.done = np.ones(s_game_amount, dtype=bool)

        game.last_position = np.zeros([s_game_amount, 2])
        return

    # testing the AI with new games
    def model_validation(self):
        r_testing = True
        steps = 0

        # create games
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

                    main.game_states(snake_number, r_testing)
                    steps += 1

        # all games done
        cv2.destroyAllWindows()

        return steps

    # increase difficulty based on test results
    def increase_difficulty(self, steps, e):
        avg = main.validation_reward / s_test_games
        step = steps / s_test_games

        # check limits
        step_limit = self.step_limit * s_step_limit_multiplier
        score_limit = self.step_limit * s_distance_score * s_score_limit_multiplier

        # if steps and scores are good enough increase difficulty
        if step >= step_limit and avg >= score_limit and e != 1:
            self.step_limit += s_step_increase

            # update graf limits
            step_limit = self.step_limit * s_step_limit_multiplier
            score_limit = self.step_limit * s_distance_score * s_score_limit_multiplier

            # increase random point
            game.add_len += s_add_len

            # increase exploration
            DQNA.epsilon += s_epsilon_increase
            if DQNA.epsilon > 1:
                DQNA.epsilon = 1

        # save test results for graf
        info.step_limit = step_limit
        info.score_limit = score_limit
        info.avg_scores.append(avg)
        info.avg_step.append(step)
        info.episodes.append(e)
        info.last_step = step
        info.last_score = avg

        return

if __name__ == '__main__':
    # initialize
    input_shape = np.zeros(s_state_size)
    DQNA = DQNAgent(input_shape)
    info = INFO(tf)
    main = MAIN()
    game = GAME()

    start = time.time()
    # define how many episodes (not games, all games with one step)
    for e in tqdm(range(1, s_episodes + 1), ascii=True, unit=' episodes'):
        r_testing = False
        # make multiple games at once
        for snake_number in range(s_game_amount):
            # create a new game if needed
            if game.done[snake_number]:
                game.spawn_snake(snake_number)
                game.spawn_apple(snake_number)

            # all game states
            main.game_states(snake_number, r_testing)

        # Train model
        DQNA.train_model(e)

        # test the model
        if s_test_model:
            if e % s_test_rate == 0 or e == 1:
                # reset values for testing
                main.reset()

                # model validation
                steps = main.model_validation()

                # increase difficulty
                main.increase_difficulty(steps, e)

                # reset values for training
                main.reset()

                # check graf lengths FIX THESE
                info.balance_episodes()
                info.balance_steps()
                info.balance_scores()

                # draw graf
                info.print_graf(DQNA.epsilon, e, game.add_len)

        # funny timer
        if time.time() - start >= 36_000:
            DQNA.save_model(e, force=True)
            print("Times up boy", time.time()-start)
            quit()


