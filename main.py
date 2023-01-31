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

        # get closest snake
        snake_len = len(snake) - 2
        snake_len = int(snake_len)

        # max snake len, delete head and apple positions
        max_len = s_state_size
        max_len -= 4

        lengths = []
        snake_coordination = []
        snake_body = snake[2:]
        for i in range(snake_len):
            # break if all empty
            if i > 0 and np.all(snake_body[i] - 1):
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

        # display for testing
        # if snake_number == 0:
        #     background = info.draw(snake_number, game.snake)
        #     info.screen(background)

        # pick action
        action = DQNA.get_qs(state, r_testing)
        # action = int(input(": "))

        # move snake
        done = game.move_snake(action, snake_number)

        # check snake
        point, done = game.check(snake_number, done)

        # add snake even without the apple
        if np.random.rand() > 0.5 and not done and not r_testing:
            game.point[snake_number] = True

        # reward calculations
        step_reward = game.reward_calculation(point, snake_number)

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

        main.ep_reward += step_reward

        # reset steps and step limit
        if game.done[snake_number]:
            main.step[snake_number] = 0
        else:
            main.step[snake_number] += 1
            if main.step[snake_number] > 300:
                game.done[snake_number] = True
                main.step[snake_number] = 0

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
        snake_number = 0

        return snake_number

    # testing the AI with new games
    def testing_ai(self):
        r_testing = True
        steps = s_test_games

        # reset saves to testing mode
        snake_number = main.reset(s_test_games)
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
        info.print_graf(main.ep_reward, steps, DQNA.epsilon)
        # reset values to train mode
        snake_number = main.reset(s_game_amount)

        return snake_number


if __name__ == '__main__':
    # initialize
    main = main()
    game = game()
    input_shape = np.zeros(s_state_size)
    DQNA = DQNAgent(input_shape)
    info = info(tf)

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

                # game thread
                # main.game_states(snake_number, r_testing)
                game_thread = threading.Thread(target=main.game_states, args=(snake_number, r_testing,))
                game_thread.start()

            # count when all the games have ended
            games_done += np.count_nonzero(game.done)

            if threading.activeCount() < 10:
                # train thread after all the games have taken a step
                # DQNA.train_model()
                train_thread = threading.Thread(target=DQNA.train_model)
                train_thread.start()

        # episode end stuff
        main.ep_reward = 0
        if len(DQNA.replay_memory) >= s_min_memory:
            # model modifications
            DQNA.epsilon_decay()
            DQNA.target_update(e)

            if e % s_save_rate == 0:
                # save model
                game_thread.join()
                train_thread.join()
                time.sleep(0.1)
                DQNA.model.save(f'models/{s_save_model_name}_episode_{e:}.model')
                # print("")
                # print("Model saved", f'models/{s_model_name}_episode_{e:}.model')

            # test the model
            if s_testing_ai and e % s_test_rate == 0:
                game_thread.join()
                train_thread.join()
                time.sleep(0.1)
                main.testing_ai()



