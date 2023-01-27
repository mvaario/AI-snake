from game import *
from DQNAgent import *
from info import *
import numpy as np
from tqdm import tqdm
import time


class main:
    def __init__(self):
        # rewards and step
        self.step = np.zeros([s_game_amount, 1])
        self.ep_reward = 0

    # Create state for neuron network
    def create_state(self, snake_number):
        # No idea why this need to be copied, but if not game.snake will change
        snake = np.copy(game.snake[snake_number])

        # size
        size_y = s_size[0]
        size_x = s_size[1]

        # apple position
        apple = snake[0]
        y = apple[0] / size_y
        x = apple[1] / size_x
        apple = [y, x]

        # head position
        head = snake[1]
        y = head[0] / size_y
        x = head[1] / size_x
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
            # snake cordination
            k = i * 2
            coordination = snake_body[i]

            # coordination distance from snake head
            distance_1 = abs(snake[1] - coordination)
            distance = int(np.sum(distance_1))

            coordination[0] = coordination[0] / size_y
            coordination[1] = coordination[1] / size_x

            # save distances and coordination
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

        state = np.concatenate((apple, head, snake_coordination))

        while len(state) < s_state_size:
            state = np.append(state, 0)

        return state

    # game states
    def game_states(self, snake_number, r_testing):
        if game.done[snake_number]:
            return

        # save state
        state = main.create_state(snake_number)

        # pick action
        action = DQNA.get_qs(state, r_testing)

        # display for testing
        # if snake_number == 0:
        # background = info.draw(snake_number, game.snake)
        # info.screen(background)

        # move snake
        done = game.move_snake(action, snake_number)

        # check snake
        point = game.check(snake_number, done)

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
                                      game.done[snake_number, 0]
                                      )

        main.ep_reward += step_reward

        # reset steps and step limit
        if game.done[snake_number]:
            main.step[snake_number] = 0
        else:
            main.step[snake_number] += 1
            if main.step[snake_number] > 500 and not r_testing:
                game.done[snake_number] = True
                main.step[snake_number] = 0

        return

    # reset all for testing
    def reset(self, games):
        # testing uses different number of games
        self.step = np.zeros([games, 1])
        self.ep_reward = 0

        game.snake = np.zeros([games, s_size[0] * s_size[1], 2])
        game.snake_old = np.zeros([games, 1, 2])
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
            game.done[snake_number] = False

        # play all the games one time
        while not np.all(game.done):
            # all games one step
            for snake_number in range(s_test_games):
                main.game_states(snake_number, r_testing)

                # show game
                if snake_number == 0:
                    if not game.done[snake_number]:
                        background = info.draw(snake_number, game.snake)
                        info.screen(background)

                if not game.done[snake_number]:
                    steps += 1

        # all games done
        cv2.destroyAllWindows()
        info.scores.append(main.ep_reward)
        info.print_graf(steps, DQNA.epsilon)
        # reset saves to train mode
        snake_number = main.reset(s_game_amount)

        return snake_number

if __name__ == '__main__':
    # initialize
    main = main()
    game = game()
    info = info()

    input_shape = np.zeros(s_state_size)
    DQNA = DQNAgent(input_shape)

    # check if tensorflow uses GPU or CPU
    print("")
    if len(tf.config.list_physical_devices('GPU')) == 1:
        print("Tensorflow using GPU")
    else:
        print("Tensorflow using CPU")
    print("")


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
                    game.done[snake_number] = False

                # make a thread for game
                # main.game_states(snake_number, r_testing)
                game_thread = threading.Thread(target=main.game_states, args=(snake_number, r_testing,))
                game_thread.start()

            # make training thread
            # DQNA.train_model()
            train_thread = threading.Thread(target=DQNA.train_model)
            train_thread.start()
            # count when all the games have ended
            games_done += np.count_nonzero(game.done)

        # episode end stuff
        info.scores.append(main.ep_reward)
        main.ep_reward = 0
        if len(DQNA.replay_memory) > s_min_memory:
            if s_testing_ai:
                if e % s_test_rate == 0 or e == 1:
                    game_thread.join()
                    train_thread.join()
                    time.sleep(0.1)
                    main.testing_ai()

            # epsilon decay
            if DQNA.epsilon > s_epsilon_min:
                DQNA.epsilon *= s_epsilon_decay
                DQNA.epsilon = max(s_epsilon_min, DQNA.epsilon)

            # update target model
            if e % s_update_rate == 0:
                DQNA.target_model.set_weights(DQNA.model.get_weights())


            # save model
            if e % s_save_rate == 0:
                DQNA.model.save(f'models/{s_model_name}_episode_{e:}.model')
                print("")
                print("Model saved", f'models/{s_model_name}_episode_{e:}.model')

