# Apollo 11 Project - LunarLander

import gym
import numpy as np
import pandas as pd
from collections import deque
import random


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.models import load_model

import pickle
from matplotlib import pyplot as plt
import tensorflow as tf
with tf.device('/DML:0'):
    class DQN:
        def __init__(self, env, lr, gamma, epsilon, epsilon_decay):

            self.env = env
            self.action_space = env.action_space
            # defines the structure of the observations your environment will be returning
            self.observation_space = env.observation_space
            self.counter = 0

            # viteza de actualizare a estimarilor
            self.lr = lr
            # discount factor
            self.gamma = gamma
            # probabilitatea ca sa alegem o actiune random
            self.epsilon = epsilon
            # epsilon va scadea putin cate putin ca se se faca actiuni random, cu 0.05
            self.epsilon_decay = epsilon_decay
            self.rewards_list = []

            self.replay_memory_buffer = deque(maxlen=500000)
            self.batch_size = 64
            # sa nu scada probabilitatea sub 0.01
            self.epsilon_min = 0.01
            # numarul de actiuni
            self.num_action_space = self.action_space.n
            # size unei stari
            self.num_observation_space = env.observation_space.shape[0]
            self.model = self.initialize_model()

        def initialize_model(self):
            # la un strat dens , output-ul este = activation_fct(inp*weights + bias)
            # cu w, b generate deja si bias mi se pare ca e optional daca este un param setat pe true
            model = Sequential()
            model.add(Dense(256, input_dim=self.num_observation_space, activation=relu))
            model.add(Dense(128, activation=relu))
            model.add(Dense(self.num_action_space, activation=linear))

            # Compile the model
            model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr), metrics=['acc'])
            print(model.summary())
            return model

        def choose_action(self, state):
            if np.random.rand() < self.epsilon:
                return random.randrange(self.num_action_space)

            predicted_actions = self.model.predict(state)
            return np.argmax(predicted_actions[0])

        def add_to_replay_memory(self, state, action, reward, next_state, done):
            self.replay_memory_buffer.append((state, action, reward, next_state, done))

        def learn_and_update_weights_by_reply(self):
            # Check replay_memory_buffer size
            if len(self.replay_memory_buffer) < self.batch_size or self.counter != 0:
                return

            # Implement early stopping
            if np.mean(self.rewards_list[-10:]) > 180:
                return

            random_sample = self.get_random_sample_from_replay_mem()
            states, actions, rewards, next_states, done_list = self.get_attribues_from_sample(random_sample)

            targets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - done_list)
            # print("Targets ", targets)
            target_vec = self.model.predict_on_batch(states)
            # print("Target vec ", target_vec)
            indexes = np.array([i for i in range(self.batch_size)])
            # print("Indecsi ", indexes)
            target_vec[indexes, actions] = targets
            # print("Target vec 2 ", target_vec)
            # print("tip target_vec", target_vec.shape())

            # antreneaza reteaua pentru sample-ul selectat din buffer replay
            self.model.fit(states, target_vec, epochs=1, verbose=0)

        # Get info from sample
        def get_attribues_from_sample(self, random_sample):
            states = np.array([i[0] for i in random_sample])
            actions = np.array([i[1] for i in random_sample])
            rewards = np.array([i[2] for i in random_sample])
            next_states = np.array([i[3] for i in random_sample])
            done_list = np.array([i[4] for i in random_sample])
            states = np.squeeze(states)
            next_states = np.squeeze(next_states)
            return np.squeeze(states), actions, rewards, next_states, done_list

        # Alege din replay un batch de size 64
        def get_random_sample_from_replay_mem(self):
            random_sample = random.sample(self.replay_memory_buffer, self.batch_size)
            return random_sample
        # Antrenam modelul pentru 500 de episoade
        def train(self, num_episodes=500, can_stop=True):
            for episode in range(num_episodes):
                state = env.reset()
                reward_for_episode = 0
                # per episod naveta face maxim 5000 de pasi(timestamps/frames)
                num_steps = 5000
                state = np.reshape(state, (1, self.num_observation_space))
                for step in range(num_steps):
                    # Afiseaza(rendeaza) jocul
                    # env.render()
                    received_action = self.choose_action(state)
                    # print("received_action:", received_action)
                    next_state, reward, done, info = env.step(received_action)
                    next_state = np.reshape(next_state, [1, self.num_observation_space])
                    # Store the experience in replay memory
                    self.add_to_replay_memory(state, received_action, reward, next_state, done)
                    # Add up rewards
                    reward_for_episode += reward
                    state = next_state
                    self.update_counter()
                    self.learn_and_update_weights_by_reply()

                    if done:
                        break
                self.rewards_list.append(reward_for_episode)

                # Decay the epsilon after each experience completion
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                # Check for breaking condition
                last_rewards_mean = np.mean(self.rewards_list[-100:])
                if last_rewards_mean > 200 and can_stop:
                    print("DQN Training Complete...")
                    break
                print(episode, "\t: Episode || Reward: ",reward_for_episode, "\t|| Average Reward: ",last_rewards_mean, "\t epsilon: ", self.epsilon )

        # Din 5 in 5 pasi, se extrage un sample din replay buffer pentru antrenare
        def update_counter(self):
            self.counter += 1
            step_size = 5
            self.counter = self.counter % step_size

        # salvam modelul
        def save(self, name):
            self.model.save(name)

    # Testam modelul deja antrenat
    def test_already_trained_model(trained_model):
        rewards_list = []
        num_test_episode = 200
        env = gym.make("LunarLander-v2")
        print("Starting Testing of the trained model...")

        #The timestamps
        step_count = 5000

        for test_episode in range(num_test_episode):
            current_state = env.reset()
            num_observation_space = env.observation_space.shape[0]
            current_state = np.reshape(current_state, [1, num_observation_space])
            reward_for_episode = 0
            for step in range(step_count):
                env.render()
                selected_action = np.argmax(trained_model.predict(current_state)[0])
                new_state, reward, done, info = env.step(selected_action)
                new_state = np.reshape(new_state, [1, num_observation_space])
                current_state = new_state
                reward_for_episode += reward
                if done:
                    break
            rewards_list.append(reward_for_episode)
            print(test_episode, "\t: Episode || Reward: ", reward_for_episode)

        return rewards_list

    def plot_df(df, chart_name, title, x_axis_label, y_axis_label):
        plt.rcParams.update({'font.size': 17})
        df['rolling_mean'] = df[df.columns[0]].rolling(100).mean()
        plt.figure(figsize=(15, 8))
        #plt.close()

        plt.figure()
        # plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
        plot = df.plot(linewidth=1.5, figsize=(15, 8))
        plot.set_xlabel(x_axis_label)
        plot.set_ylabel(y_axis_label)
        # plt.ylim((-400, 300))
        fig = plot.get_figure()
        plt.legend().set_visible(True)
        fig.savefig(chart_name)
        plt.show()


    def plot_df2(df, chart_name, title, x_axis_label, y_axis_label):
        df['mean'] = df[df.columns[0]].mean()
        plt.rcParams.update({'font.size': 17})
        plt.figure(figsize=(15, 8))
        plt.close()
        plt.figure()
        plot = df.plot(linewidth=1.5, figsize=(15, 8), title=title)
        plot = df.plot(linewidth=1.5, figsize=(15, 8))
        plot.set_xlabel(x_axis_label)
        plot.set_ylabel(y_axis_label)
        plt.ylim((0, 300))
        plt.xlim((0, 100))
        plt.legend().set_visible(True)
        fig = plot.get_figure()
        fig.savefig(chart_name)
        plt.show()


    if __name__ == '__main__':
        env = gym.make('LunarLander-v2')

        # set seeds
        #  La rulari diferite sa fie aceeasi  prima stare  dar de la un episod la altul se schimba
        env.seed(0)
        #  la rulari diferite o sa fie generate aceleasi numere random
        # sa fie asigurate aceleasi conditii
        np.random.seed(0)

        # setting up params
        lr = 0.001
        epsilon = 1.0
        # va scadea putin cate putin probabilitatea sa fie alese actiuni random, mai exact cu 0.005
        epsilon_decay = 0.995
        # The discount factor
        gamma = 0.99
        training_episodes = 500
        print('Informations about the model')
        model = DQN(env, lr, gamma, epsilon, epsilon_decay)
        model.train(training_episodes, True)

        # Save Everything
        save_dir = "saved_models"
        # Save trained model
        model.save(save_dir + "trained_model.h5")

        # Save Rewards list
        pickle.dump(model.rewards_list, open(save_dir + "train_rewards_list.p", "wb"))
        rewards_list = pickle.load(open(save_dir + "train_rewards_list.p", "rb"))

        # plot reward in graph
        reward_df = pd.DataFrame(rewards_list)
        plot_df(reward_df, "Figure 1: Reward for each training episode", "Reward for each training episode", "Episode",
                "Reward")

        # Test the model
        trained_model = load_model(save_dir + "trained_model.h5")
        test_rewards = test_already_trained_model(trained_model)
        pickle.dump(test_rewards, open(save_dir + "test_rewards.p", "wb"))
        test_rewards = pickle.load(open(save_dir + "test_rewards.p", "rb"))

        plot_df2(pd.DataFrame(test_rewards), "Figure 2: Reward for each testing episode",
                 "Reward for each testing episode", "Episode", "Reward")
        print("Training and Testing Completed...!")
