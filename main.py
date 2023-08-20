import threading
from game import *
from DQNAgent import *
from info import *
import numpy as np
from tqdm import tqdm
import time
import tensorflow as tf


class main:
    def __init__(self):
        # define state
        self.ori_state = np.zeros(s_state_size)
        # rewards and step
        self.step = np.zeros([s_game_amount, 1])
        self.ep_reward = 0
        self.step_limit = s_step_limit

    # Create state for neuron network
    def create_state(self, snake_number):
        # No idea why this need to be copied, but if not game.snake will change
        snake = np.copy(game.snake[snake_number])

        # apple position
        apple = snake[0]
        y = apple[0] / s_size[0]
        x = apple[1] / s_size[1]
        apple = [y, x]

        # head position
        head = snake[1]
        y = head[0] / s_size[0]
        x = head[1] / s_size[1]
        head = [y, x]

        # max snake len, delete head and apple positions
        max_len = s_state_size
        max_len -= 4

        lengths = []
        snake_coordination = []
        snake_body = snake[2:]
        for i in range(len(snake_body)):
            # break if all empty
            if i > 0 and np.all(snake_body[i] == - 1):
                break

            # snake coordination
            coordination = snake_body[i]

            # coordination distance from snake head
            distance_1 = abs(snake[1] - coordination)
            distance = int(np.sum(distance_1))

            coordination[0] = coordination[0] / s_size[0]
            coordination[1] = coordination[1] / s_size[1]

            # save distances and coordination
            k = i * 2
            if k < max_len:
                lengths.append(distance)
                snake_coordination.append(coordination)

            # if snake is too long, save the closest
            else:
                max = np.max(lengths)
                if distance < max:
                    for l in range(len(lengths)):
                        if lengths[l] == max:
                            lengths[l] = distance
                            snake_coordination[l][0] = coordination[0]
                            snake_coordination[l][1] = coordination[1]
                            break

        snake_coordination = np.array(snake_coordination)
        snake_coordination = np.reshape(snake_coordination, (-1))

        state = np.copy(self.ori_state)
        state[0] = apple[0]
        state[1] = apple[1]
        state[2] = head[0]
        state[3] = head[1]
        for i in range(len(snake_coordination)):
            k = i + 4
            state[k] = snake_coordination[i]

        return state

    # game states
    def game_states(self, snake_number, r_testing):
        if game.done[snake_number]:
            return

        # save state
        state = main.create_state(snake_number)

        # pick action
        action = DQNA.get_qs(state, r_testing)
        # action = np.random.randint(1,3)

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
            next_state = main.create_state(snake_number)

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

        game.snake = np.ones([games, s_size[0] * s_size[1], 2])
        game.snake = np.negative(game.snake)
        game.last_position = np.zeros([games, 2])
        game.point = np.zeros([games, 1], dtype=bool)
        game.done = np.ones([games, 1], dtype=bool)

        return

    # testing the AI with new games
    def testing_ai(self, e):
        r_testing = True
        steps = s_test_games

        # reset saves to testing mode
        main.reset(s_test_games)
        for snake_number in range(s_test_games):
            game.spawn_snake(snake_number)
            game.spawn_apple(snake_number)

        # play all the games one time
        while not np.all(game.done):
            # one step every game
            for snake_number in range(s_test_games):
                main.game_states(snake_number, r_testing)

                # show game
                if snake_number == 0:
                    if not game.done[snake_number]:
                        background = info.draw(snake_number, game.snake)
                        info.screen(background)
                    else:
                        cv2.destroyAllWindows()
                if not game.done[snake_number]:
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
            main.step_limit += 5

            # increase random point
            game.random_poit += 0.05
            if game.random_poit >= 0.75:
                game.random_poit = 0.75

            # decay distance score
            game.distance_score -= 0.05
            if game.distance_score <= 0:
                game.distance_score = 0

            # increase exploration
            DQNA.epsilon += 0.25
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
            DQNA.model.save(f'models/{s_save_model_name}_episode_{e:}.model')
            # print("")
            # print("Model saved", f'models/{s_model_name}_episode_{e:}.model')

        # # test the model
        if s_testing_ai:
            if e % s_test_rate == 0 or e <= 1:
                time.sleep(0.1)
                main.testing_ai(e)
