import cv2
from game import *
from DQNAgent import *
from tqdm import tqdm
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import time
import matplotlib.pyplot as plt

class main:
    def __init__(self):
        self.reward = 0
        self.full_reward = 0
        self.step = 0

        # logging
        self.ten_round_reward = 0
        self.episodes = []
        self.scores = []
        self.epsilon = []
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

        max_len = state_size - 4

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
                coordination = np.array([snake_1, snake_2])

                # save distances and coordination
                if k < max_len:
                    lengths = np.append(lengths, distance)
                    snake_coordination = np.append(snake_coordination, coordination)
                # if snake is too long save the closest
                else:
                    max = np.max(lengths)
                    if distance < max:
                        for l in range(len(lengths)):
                            if lengths[l] == max:
                                lengths[l] = distance
                                snake_coordination[l] = coordination[0]
                                snake_coordination[l+1] = coordination[1]

        while len(snake_coordination) < max_len:
            snake_coordination = np.append(snake_coordination, 1)

        state = np.array([apple, head])
        state = np.append(state, snake_coordination)
        return state

    def reward_calculation(self, done, point):
        if done:
            self.reward -= penalty
        elif point:
            self.reward += apple_score
        else:
            distance = abs(game.apple - game.head)
            back = np.array([game.snake[0], game.snake[1]])
            old_dis = abs(game.apple - back)
            difference = old_dis - distance
            self.reward += difference[0] * distance_score
            self.reward += difference[1] * distance_score

        return

    def screen(self, background):
        # show screen
        game = cv2.resize(background, (500, 500), interpolation=cv2.INTER_NEAREST)
        game = np.uint8(game)
        cv2.imshow("game", game)
        cv2.moveWindow("game", -520, 40)
        cv2.waitKey(wait_time)
        return

    def finish(self,e, step, start):
        # calculate avg reward / epsilon
        main.full_reward += ep_reward
        avg_reward = main.full_reward / e
        # calculate avg step / epsilon
        main.step += step
        avg_step = main.step / e

        # print the graf
        if logging and len(DQNA.replay_memory) > min_memory:
            self.episodes.append(e)
            self.scores.append(avg_reward)
            self.epsilon.append(DQNA.epsilon)

            plt.xlabel("Episode")
            plt.ylabel("Score / Epsilon")

            plt.plot(self.episodes, self.epsilon, label='Epsilon')
            plt.plot(self.episodes, self.scores, label='Scores')

            plt.legend()
            plt.show(block=False)
            plt.pause(0.0000000000001)
            plt.cla()

        # saving and printing
        if e % save_rate == 0:
            loop_time = round(time.time() - start, 2)
            loop_time = round(loop_time / 60, 2)
            start = time.time()
            print("")
            print("Round", e,
                  "Epsilon:", round(DQNA.epsilon, 3),
                  "Episode time", loop_time,
                  "Avg step", round(avg_step, 2),
                  "Avg reward", round(avg_reward, 2)
                  )
            # save model
            if save_model and train:
                DQNA.model.save(f'models/{model_name}_episode_{e:}_avg_{round(avg_reward, 2):}.model')

        return start

if __name__ == '__main__':
    main = main()
    game = game()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    input_shape = np.zeros(state_size)
    # input_shape = np.expand_dims(input_shape, -1)
    DQNA = DQNAgent(input_shape)
    # start timer
    start = time.time()

    # define episodes
    for e in tqdm(range(1, n_episodes + 1), ascii=True, unit='episodes'):
        # count reward
        ep_reward = 0
        # starting step
        step = 0

        # create a new game
        game.spawn_snake()
        game.spawn_apple()

        # point = got the apple / done = dead
        done = False
        point = False
        # in game
        while not done:
            # create state
            # state, direction = main.create_state()
            state = main.create_state()

            # pick action
            action = DQNA.get_qs(state)

            # move snake
            game.move_snake(action)

            # check snake
            done, point = game.check()

            # reward calculations
            main.reward_calculation(done, point)

            if step >= step_limit:
                done = True
            step += 1

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

            if train and step > step_min:
                # update memory
                DQNA.update_replay_memory(state, action, main.reward, next_state, done)

                # train model
                DQNA.train_model(e)

            # episode reward
            ep_reward += main.reward
            main.reward = 0

        # epsilon decay
        if DQNA.epsilon > epsilon_min:
            DQNA.epsilon *= epsilon_decay
            DQNA.epsilon = max(epsilon_min, DQNA.epsilon)

        # printing and logging
        start = main.finish(e,step, start)


