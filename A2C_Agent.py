import copy
import random
import numpy as np
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Activation
from keras.models import Sequential
from keras.optimizers import Adam


class A2C_Agent:

    def __init__(self, state_size, action_size, test_mode=False):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.test_mode = test_mode
        self.buffer_size = self.state_size[3]
        self.gamma = 0.9  # discount factor

        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.1
        self.learning_rate = 0.00025
        self.buffer = np.zeros((1, self.state_size[0], self.state_size[1], self.state_size[2], self.buffer_size))

        self.actor = self._build_actor()
        self.critic = self._build_critic()

        if self.test_mode is True:
            self.epsilon = 0.01
        else:
            self.epsilon = 1

    def _build_actor(self):
        model = Sequential()

        model.add(Conv3D(32, kernel_size=(5, 5, 5), padding="same", input_shape=self.state_size))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(32, kernel_size=(5, 5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(units=512, activation="tanh"))
        model.add(Dense(units=256, activation="tanh"))
        model.add(Dense(units=128, activation="tanh"))
        # model.add(Dense(units=64, activation="tanh"))
        model.add(Dense(units=self.action_size, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def _build_critic(self):
        model = Sequential()

        model.add(Conv3D(32, kernel_size=(5, 5, 5), padding="same", input_shape=self.state_size))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(32, kernel_size=(5, 5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
        model.add(Activation("tanh"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        model.add(Dense(units=512, activation="relu"))
        model.add(Dense(units=256, activation="relu"))
        model.add(Dense(units=128, activation="linear"))
        # model.add(Dense(units=64, activation="tanh"))
        model.add(Dense(units=self.value_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def act(self, state, test_state=False):

        # The stack of 4 frames is used as a single state

        for i in range(self.buffer_size):
            if i != self.buffer_size - 1:
                self.buffer[:, :, :, :, i] = self.buffer[:, :, :, :, i + 1]  # shift the frames
            else:
                self.buffer[:, :, :, :, self.buffer_size - 1] = state  # add a new frame

        policy = self.actor.predict(self.buffer)[0]

        if test_state is True:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size), policy
            else:
                return np.argmax(policy), policy
        else:
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size), policy

            return np.random.choice(self.action_size, 1, p=policy)[0], policy  # The second argument is policy vector

    def flush_buffer(self):
        self.buffer = np.zeros((1, self.state_size[0], self.state_size[1], self.state_size[2], self.buffer_size))

    def replay(self, state, action, reward, next_state):

        buffer_state = copy.deepcopy(self.buffer)  # last frame is the state
        buffer_next_state = copy.deepcopy(self.buffer)
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(buffer_state)[0]
        for i in range(self.buffer_size):
            if i != self.buffer_size - 1:
                buffer_next_state[:, :, :, :, i] = buffer_next_state[:, :, :, :, i + 1]  # shift the frames
            else:
                buffer_next_state[:, :, :, :, self.buffer_size - 1] = next_state  # add a new frame

        next_value = self.critic.predict(buffer_next_state)[0]  # last frame is the next state

        advantages[0][action] = reward + self.gamma * (next_value) - value
        target[0][0] = reward + self.gamma * next_value

        self.actor.fit(buffer_state, advantages, epochs=1, verbose=0)
        self.critic.fit(buffer_state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
