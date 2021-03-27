import cv2
from DQNAgent import *
from settings import *
from tqdm import tqdm
import time
import pygame as pg
import math
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class main:
    def __init__(self):
        self.head = 0
        self.snake = 0
        self.apple = 0

        self.old_snake = []

        self.reward = 0
        self.ep_rewards = []

        self.distance = 0
        self.old_distance = 0
        self.min_distance = 0

        self.avg_reward = 0
        self.min_reward = 0

    def background_head(self):
        background = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        head = np.random.randint(3, size[1] - 4, size=2)

        self.head = head
        return background

    def spawn_snake(self):
        snake = np.array([-1, -2])
        while np.any(1 > snake) or np.any(snake > (size[0] - 2)):
            snake = np.random.randint(0, 4)
            if snake == 0:
                snake = np.array([self.head[0] - 1, self.head[1]])
            elif snake == 1:
                snake = np.array([self.head[0] + 1, self.head[1]])
            elif snake == 2:
                snake = np.array([self.head[0], self.head[1] - 1])
            else:
                snake = np.array([self.head[0], self.head[1] + 1])

        self.snake = snake

        return

    def spawn_apple(self):
        apple = np.random.randint(1, size[0] - 2, size=2)
        apple = main.check_apple(apple)
        while np.all(apple == self.head) or np.all(apple == self.snake):
            apple = np.random.randint(1, size[0] - 2, size=2)
            apple = main.check_apple(apple)

        self.apple = apple
        return

    def check_apple(self, apple):
        l = int(len(self.old_snake) / 2)
        for i in range(l):
            x = i * 2
            if apple[0] == self.old_snake[x] and apple[1] == self.old_snake[x + 1]:
                apple = self.head
        return apple

    def movement(self, action):
        # 0 = ylös
        # 1 = alas
        # 2 = oikea
        # 3 = vasen

        if action == 0:
            direction = [-1, 0]
        elif action == 1:
            direction = [1, 0]
        elif action == 2:
            direction = [0, 1]
        elif action == 3:
            direction = [0, -1]
        else:
            direction = [0, 0]

        head = self.head + direction

        if np.any(head <= 0):
            head = self.head
            done = True
        elif np.any(head >= (size[0] - 1)):
            head = self.head
            done = True
        elif np.all(head == self.snake):
            head = self.head
            done = True
        else:
            done = False
            l = int(len(self.old_snake) / 2)
            for i in range(l):
                x = i * 2
                if head[0] == self.old_snake[x] and head[1] == self.old_snake[x + 1]:
                    head = self.head
                    done = True

        if np.any(self.head != head):
            x = int(len(self.old_snake) - 1)
            l = int(len(self.old_snake) / 2 - 1)
            for i in range(l):
                y = x - (i * 2)
                self.old_snake[y] = self.old_snake[y - 2]
                self.old_snake[y - 1] = self.old_snake[y - 3]

            if len(self.old_snake) != 0:
                self.old_snake[0] = self.snake[0]
                self.old_snake[1] = self.snake[1]
            self.snake = self.head

        self.head = head

        return done

    def update(self, background):
        background[:, :, :] = 0
        background[0, :] = gray
        background[:, 0] = gray
        background[size[1] - 1, :] = gray
        background[:, size[0] - 1] = gray

        background[self.apple[0], self.apple[1], :] = red
        background[self.head[0], self.head[1], :] = green
        background[self.snake[0], self.snake[1], :] = gray


        l = int(len(self.old_snake) / 2)
        for i in range(l):
            x = i * 2
            background[int(self.old_snake[x]), int(self.old_snake[x + 1]), :] = gray

        if np.all(self.head == self.apple):
            main.spawn_apple()
            background[self.apple[0], self.apple[1], :] = red
            self.reward += apple_score
            self.old_snake = np.append(self.old_snake, self.snake)
            self.old_snake = np.array(self.old_snake)
            self.min_distance = 0

        return background

    def creating_state(self):
        next_state = np.zeros((size[0], size[1]))
        next_state[0, :] = -1
        next_state[:, 0] = -1
        next_state[size[1] - 1, :] = -1
        next_state[:, size[0] - 1] = -1

        next_state[self.head[0], self.head[1]] = 1
        next_state[self.snake[0], self.snake[1]] = -1
        if len(self.apple) != 0:
            next_state[self.apple[0], self.apple[1]] = 2

        for i in range(int(len(self.old_snake) / 2)):
            x = i * 2
            next_state[int(self.old_snake[x]), int(self.old_snake[x + 1])] = -1

        next_state = np.reshape(next_state, (size[0], size[1]))
        next_state = np.expand_dims(next_state, -1)

        return next_state

    def screen(self, background):
        game = cv2.resize(background, (500, 500), interpolation=cv2.INTER_NEAREST)
        game = np.uint8(game)
        cv2.imshow("game", game)
        cv2.moveWindow("game", -600, 40)
        return

    def reward_calculation(self, done):
        if done:
            self.reward -= penalty
            self.min_distance = 0
            self.old_distance = 0
        else:
            # r = len(self.old_snake) / 2
            # r = r / 10
            # r = round(r)
            # self.reward += r

            dis = (self.apple - self.head)
            dis = abs(dis)
            dis = dis[0]**2 + dis[1]**2
            dis = math.sqrt(dis)

            if self.min_distance == 0:
                self.min_distance = dis
                self.old_distance = dis

            if dis < self.min_distance:
                self.reward += 1
                self.min_distance = dis

            elif dis > self.old_distance != 0:
                self.reward -= 1

            self.old_distance = dis


        return


    def get_inputs(self):
        keys = pg.key.get_pressed()

        if keys[pg.K_RIGHT]:
            action = 2
        elif keys[pg.K_LEFT]:
            action = 3
        elif keys[pg.K_UP]:
            action = 0
        elif keys[pg.K_DOWN]:
            action = 1
        else:
            action = 10

        return action


if __name__ == '__main__':
    main = main()

    if play:
        pg.init()
        window = pg.display.set_mode((30, 30))
    else:
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.gpu_options.allow_growth = False
        session = InteractiveSession(config=config)

        input_shape = np.zeros((size[0], size[1]))
        input_shape = np.expand_dims(input_shape, -1)
        DQNA = DQNAgent(input_shape)

    for e in tqdm(range(1, n_episodes + 1), ascii=True, unit='episodes'):
        background = main.background_head()
        main.spawn_snake()
        main.spawn_apple()
        background = main.update(background)

        ep_reward = 0
        done = False
        main.old_snake = []
        step = 0
        while not done:
            state = main.creating_state()
            if play:
                action = main.get_inputs()
            else:
                r = main.snake - main.head
                action = DQNA.get_qs(state, r, step)


            done = main.movement(action)
            background = main.update(background)

            main.reward_calculation(done)
            if train and step >= step_limit:
                done = True
            step += 1

            if show:
                if train and e % show_rate == 0:
                    main.screen(background)
                    cv2.waitKey(1)
                elif not train:
                    main.screen(background)
                    cv2.waitKey(20)
                else:
                    cv2.destroyAllWindows()

            if train and step > train_step:
                    next_state = main.creating_state()

                    DQNA.update_replay_memory(state, action, main.reward, next_state, done)
                    DQNA.train_2_model(e)

            ep_reward += main.reward
            # if main.reward != 0:
            #     print(main.reward)
            main.reward = 0




        if train:
            if DQNA.epsilon > epsilon_min and step > train_step:
                DQNA.epsilon *= epsilon_decay
                DQNA.epsilon = max(epsilon_min, DQNA.epsilon)

            ep_rewards.append(ep_reward)
            # if e % 500 == 0:
            #     step_limit += 2
            #     if e % 1000:
            #         train_step += 1

            if e % 50 == 0:
                average_reward = sum(ep_rewards[-save_rate:]) / len(ep_rewards[-save_rate:])
                min_reward = min(ep_rewards[-save_rate:])
                max_reward = max(ep_rewards[-save_rate:])

                print(" Epsilon:", round(DQNA.epsilon, 3), "Avg:", round(average_reward), "Min", min_reward, "Train_step", train_step)

                if e % save_rate == 0:
                    DQNA.model.save(f'models/{model_name}_episode_{e:}_avg_{round(average_reward):}.model')
                    # print("Saved: Epsilon:", round(DQNA.epsilon, 3), "Average reward:", round(average_reward), "Min reward", min_reward)



