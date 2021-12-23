import cv2
import numpy as np
from game import *
from DQNAgent import *
from tqdm import tqdm
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import time

class main:
    def __init__(self):
        self.reward = 0
        self.full_reward = 0
        self.step = 0

    def create_state(self):
        # create background with walls and snake
        # dtype = np.uint8
        background = np.ones((size[0] + 2, size[1] + 2), dtype=np.int32)
        background[1:size[0] + 1, 1:size[1] + 1] = 0
        background[game.apple[0]+1, game.apple[1]+1] = 10
        background[game.head[0]+1, game.head[1]+1] = 5

        for i in range(game.snake_len):
            x = i * 2
            background[game.snake[x]+1, game.snake[x + 1]+1] = 1

        # snake direction
        back = np.array([game.snake[0], game.snake[1]])

        direction = game.head - back
        x = input_shape.shape[1]
        y = input_shape.shape[0]
        y_half = int((y-1) / 2)
        x_half = int((x-1) / 2)

        # create state to ai
        if direction[0] == 0:
            if direction[1] == 1:
                x_0 = game.head[1]
                x_1 = x_0 + x
            else:
                x_0 = game.head[1] - x + 3
                x_1 = game.head[1] + 3
            y_0 = game.head[0] - y_half + 1
            y_1 = game.head[0] + y_half + 2

        else:
            if direction[0] == 1:
                y_0 = game.head[0]
                y_1 = game.head[0] + y
            else:
                y_0 = game.head[0] - y + 3
                y_1 = game.head[0] + 3
            x_0 = game.head[1] - x_half + 1
            x_1 = game.head[1] + x_half + 2


        # check if state if outside of background
        if x_1 > size[1] + 2:
            x_1 = size[1] + 2
            x_0 = x_1 - x
        elif x_0 < 0:
            x_0 = 0
            x_1 = x
        if y_0 < 0:
            y_0 = 0
            y_1 = y
        elif y_1 > size[0] + 2:
            y_1 = size[0] + 2
            y_0 = y_1 - y

        state = background[y_0:y_1, x_0:x_1]
        state = state / 10

        # apple_dir = game.apple - game.head
        # state[0,0] = apple_dir[0]
        # state[0,1] = apple_dir[1]

        state = np.expand_dims(state, -1)
        return state, direction

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

if __name__ == '__main__':
    main = main()
    game = game()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    input_shape = np.zeros((input_shape[0], input_shape[1]))
    input_shape = np.expand_dims(input_shape, -1)
    DQNA = DQNAgent(input_shape)

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
            state, direction = main.create_state()

            # pick action
            action, movement = DQNA.get_qs(state, direction)
            # action = int(input("pres"))

            # check if movement = backward == dead
            done = game.check_back(movement, direction)

            # move snake
            game.move_snake(movement)

            if not done:
                # check action
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
            next_state, direction = main.create_state()

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

        # calculate avg reward / epsilon
        main.full_reward += ep_reward
        avg_reward = main.full_reward / e
        # calculate avg step / epsilon
        main.step += step
        avg_step = main.step / e

        if e % display_rate == 0:
            print("")
            print("Round", e, "Epsilon:", round(DQNA.epsilon, 3), "Avg step", round(avg_step, 2), "Avg reward", round(avg_reward, 2))

        if save_model and train:
            if e % save_rate == 0:
                DQNA.model.save(f'models/{model_name}_episode_{e:}_avg_{round(avg_reward, 2):}.model')
                print("Saved: Epsilon:", round(DQNA.epsilon, 3))

    print("loop time", round(time.time() - start, 2))


