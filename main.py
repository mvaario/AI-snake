import cv2
from game import *
from DQNAgent import *
from tqdm import tqdm
from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import time



class main:
    def __init__(self):
        self.reward = 0
        self.full_reward = 0
        self.step = 0

    def create_state(self):
        # direction
        back = np.array([game.snake[0], game.snake[1]])
        direction = game.head - back
        # walls distance
        walls = size - game.head

        # save x amount of closest snake
        snake_len = int(len(game.snake) / 2)
        lengths = []
        len_1 = input_shape.shape[0] - 1
        max_len = len_1 * input_shape.shape[1]
        for i in range(snake_len):
                # snake cordination
                k = i * 2
                snake_1 = game.snake[k]
                snake_2 = game.snake[k+1]
                coordination = np.array([snake_1, snake_2])

                # coordination distance from snake head
                distance_1 = abs(game.head - coordination)
                distance = np.sum(distance_1)

                snake_coordination = []
                # save distances and coordination
                if len(lengths) < max_len:
                    lengths.append(distance)
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
            snake_coordination = np.append(snake_coordination, 0)

        # create array for ai
        state = np.array([game.head, direction, game.apple, walls])
        state = np.append(state, snake_coordination)

        state = np.reshape(state, input_shape.shape)
        # state = np.expand_dims(state, -1)
        state = state / 50

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
    # input_shape = np.expand_dims(input_shape, -1)
    DQNA = DQNAgent(input_shape)
    # start timer
    start = time.time()
    if logging:
        webbrowser.open('http://localhost:6006/ ', new=1)
        # for check terminal: "tensorboard --logdir=logs/
        # tensorboard_callback = TensorBoard(log_dir='logs\\{}'.format(model_name))

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
            action = DQNA.get_qs(state)

            # move snake
            game.move_snake(action, direction)

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

        if e % save_rate == 0:
            loop_time = round(time.time() - start, 2)
            loop_time = round(loop_time / 60, 2)

            start = time.time()
            print("")
            print("Round", e,
                  "Epsilon:", round(DQNA.epsilon, 3),
                  save_rate, "Episode time", loop_time,
                  "Avg step", round(avg_step, 2),
                  "Avg reward", round(avg_reward, 2)
                  )
            if save_model and train:
                DQNA.model.save(f'models/{model_name}_episode_{e:}_avg_{round(avg_reward, 2):}.model')
                print("Saved: Epsilon:", round(DQNA.epsilon, 3))
