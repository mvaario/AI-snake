import cv2
import numpy as np

from game import *
from DQNAgent import *
from tqdm import tqdm
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

class main:
    def __init__(self):
        # rewards and step
        self.step_reward = 0
        self.ep_reward = 0
        self.step = 0

        # avg
        self.full_steps = 0
        self.full_reward = 0
        self.avg_step = 0
        self.avg_reward = 0

        # logging
        self.episodes = []
        self.scores = []
        self.steps = []

    def start_info(self):
        # printing stuff
        print("")
        print("--------------------------------")
        print("Load model:", load_model)
        if load_model:
            print("Model name:", load_model_name)

        print("")
        print("Train:", train)
        if train:
            print("Training setups:")
            print("\t Batch size:", batch_size)
            print("\t Update rate:", update_rate)
            print("\t Start epsilon:", start_epsilon)
            print("\t Epsilon min:", epsilon_min)
            print("\t Epsilon decay:", epsilon_decay)
            if save_rate:
                print("\t Save rate:", save_rate)

        print("")
        print("Step limit:", step_limit)
        print("Step min:", step_min)

        print("Display:", display)
        if display:
            if train:
                print("\t Display rate:", display_rate)
            else:
                print("\t Display rate:", 1)
            print("\t Wait time:", wait_time)
            print("Logging:", logging)
        print("--------------------------------")
        print("")
        return

    def create_state(self):
        # size
        size_y = size[0]
        size_x = size[1]

        # apple position
        y = game.apple[0] / size_y
        x = game.apple[1] / size_x
        apple = [y, x]

        # head position
        y = game.head[0] / size_y
        x = game.head[1] / size_x
        head = [y, x]

        # get closest snake
        snake_len = len(game.snake) / 2
        snake_len = int(snake_len)

        # max snake len
        max_len = state_size[0] * state_size[1]
        max_len -= -4

        lengths = []
        snake_coordination = []
        for i in range(snake_len):
                # snake cordination
                k = i * 2
                snake_1 = game.snake[k]
                snake_2 = game.snake[k+1]
                coordination = np.array([snake_1, snake_2])

                # coordination distance from snake head
                distance_1 = abs(game.head - coordination)
                distance = np.sum(distance_1)

                snake_1 /= size_y
                snake_2 /= size_x
                coordination = [snake_1, snake_2]

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

        while len(state)+4 < max_len:
            state = np.append(state, 1)

        state = np.reshape(state, (-1, 4))

        return state

    def reward_calculation(self, done, point):
        if done:
            self.step_reward -= penalty
        elif point:
            self.step_reward += apple_score
        else:
            distance = abs(game.apple - game.head)
            back = np.array([game.snake[0], game.snake[1]])
            old_dis = abs(game.apple - back)
            difference = old_dis - distance
            self.step_reward += difference[0] * distance_score
            self.step_reward += difference[1] * distance_score

        return

    def screen(self, background):
        # show screen
        game = cv2.resize(background, (500, 500), interpolation=cv2.INTER_NEAREST)
        game = np.uint8(game)
        cv2.imshow("game", game)
        cv2.moveWindow("game", -520, 40)
        cv2.waitKey(wait_time)
        return

    def finish(self, e, start):
        # calculate avg reward and step
        self.full_reward += self.ep_reward
        self.full_steps += self.step
        self.avg_reward = self.full_reward / e
        self.avg_step = self.full_steps / e

        # print the graf
        if logging and len(DQNA.replay_memory) > min_memory:
            self.episodes.append(e)
            self.scores.append(self.ep_reward)
            self.steps.append(self.step)

            plt.xlabel("Episode")
            plt.ylabel("Score / Steps")

            plt.grid(True)

            # plot steps and scores
            plt.plot(self.episodes, self.steps, label='Steps')
            plt.plot(self.episodes, self.scores, label='Scores')

            # title epsilon and time
            loop_time = round(time.time() - start, 2)
            plt.title(f'Epsilon: {round(DQNA.epsilon, 5)}', loc='left')
            plt.title(f'Time: {loop_time}')
            plt.title(f'Step limit: {step_limit}', loc='right')

            plt.legend()
            plt.show(block=False)
            plt.pause(0.0000000000001)
            plt.cla()

        # saving and printing
        if e % save_rate == 0:
            print("")
            print("Round", e,
                  "Epsilon:", round(DQNA.epsilon, 3),
                  "Avg step", round(self.avg_step, 2),
                  "Avg reward", round(self.avg_reward, 2)
                  )
            # save model
            if save_model and train:
                DQNA.model.save(f'models/{model_name}_episode_{e:}_avg_{round(self.avg_reward, 2):}.model')

        start_time = time.time()
        return start_time

if __name__ == '__main__':
    main = main()
    game = game()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    if logging:
        figure(figsize=(15, 5), dpi=90, facecolor=(0.43, 0.43, 0.43), clear=True)

    input_shape = np.zeros(state_size)
    # input_shape = np.expand_dims(input_shape, -1)
    DQNA = DQNAgent(input_shape)

    # print info
    main.start_info()
    # start timer
    start_time = time.time()
    # define episodes
    # for e in tqdm(range(1, n_episodes + 1), ascii=True, unit='episodes'):
    for e in range(n_episodes):
        main.ep_reward = 0
        main.step = 0

        # create a new game
        game.spawn_snake()
        game.spawn_apple()

        # while in game
        done = False
        while not done:
            # create state
            state = main.create_state()

            # pick action
            action = DQNA.get_qs(state)

            # move snake
            game.move_snake(action)

            # check snake
            done, point = game.check()

            # reward calculations
            main.reward_calculation(done, point)

            # display the game
            if display:
                if not train:
                    background = game.draw()
                    main.screen(background)
                elif e % display_rate == 0:
                    background = game.draw()
                    main.screen(background)
                else:
                    cv2.destroyAllWindows()

            # create new state
            next_state = main.create_state()

            if main.step > step_min:
                # update memory
                DQNA.update_replay_memory(state, action, main.step_reward, next_state, done)

                # train model
                if train and done:
                    DQNA.train_model(e)

            if main.step >= step_limit:
                done = True
            main.step += 1

            # episode reward
            main.ep_reward += main.step_reward
            main.step_reward = 0

        # epsilon decay
        if DQNA.epsilon > epsilon_min:
            DQNA.epsilon *= epsilon_decay
            DQNA.epsilon = max(epsilon_min, DQNA.epsilon)

        if e % save_rate == 0:
            if main.step >= step_limit:
                step_limit += 5
                step_min += 5

        # printing and logging
        # start_time = main.finish(e, start_time)

