import random
from collections import deque
import numpy as np
from keras import backend as k
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Activation, Input, \
    Lambda, Add
from keras.models import Sequential, Model
from keras.optimizers import Adam

class Double_DQNAgent:

    def __init__(self, state_size, action_size, test_mode=False, enable_dist_estimator=False
                 , enable_dist_est_buffer=False, enable_dueling_dqn=False, enable_prioritized_replay=False):

        self.state_size = state_size
        self.action_size = action_size
        self.test_mode = test_mode
        self.enable_dist_estimator = enable_dist_estimator
        self.enable_dist_est_buffer = enable_dist_est_buffer
        self.enable_dueling_dqn = enable_dueling_dqn
        self.enable_prioritized_replay = enable_prioritized_replay
        self.buffer_size = self.state_size[3]

        self.buffer = np.zeros((1, self.state_size[0], self.state_size[1], self.state_size[2], self.buffer_size))
        self.memory_size = 50000
        self.memory = deque(maxlen=self.memory_size)
        # Once the deque is full. The oldest element in the deque is removed
        self.gamma = 0.9  # discount factor

        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1

        self.learning_rate = 0.00025

        if self.enable_prioritized_replay:
            self.next_state_buffer = np.zeros(
                (1, self.state_size[0], self.state_size[1], self.state_size[2], self.buffer_size))
            self.priority_values = deque(maxlen=self.memory_size)
            self.coeff_b = 0.5
            self.coeff_b_decay = 0.999
            self.replay_probabilities = None

        if self.enable_dueling_dqn:
            self.model = self._build_model_dueling()
            self.target_model = self._build_model_dueling()
        else:
            self.model = self._build_model()
            self.target_model = self._build_model()

        if self.enable_dist_estimator:
            self.distance_estimator = self._build_distance_estimator()

        if test_mode is True:
            self.epsilon = 0.1
        else:
            self.epsilon = 1  # epsilon-greedy (initial value)

    def _build_model(self):

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
        model.add(Dense(units=self.action_size, activation="linear"))
        model.summary()

        if self.enable_prioritized_replay:
            model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        else:
            model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def _build_model_dueling(self):
        input = Input(shape=self.state_size)
        x = Conv3D(32, kernel_size=(5, 5, 5), padding="same", activation="tanh")(input)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = Conv3D(32, kernel_size=(5, 5, 5), padding="same", activation="tanh")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = Conv3D(64, kernel_size=(5, 5, 5), padding="same", activation="tanh")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        x = Conv3D(64, kernel_size=(5, 5, 5), padding="same", activation="tanh")(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)
        input_dueling = Flatten()(x)

        state_value = Dense(512, activation="tanh", kernel_initializer='normal')(input_dueling)
        state_value = Dense(256, activation="tanh", kernel_initializer='normal')(state_value)
        state_value = Dense(128, activation="tanh", kernel_initializer='normal')(state_value)
        state_value = Dense(1, activation="linear", kernel_initializer='normal')(state_value)
        state_value = Lambda(lambda s: k.expand_dims(s[:, 0], -1), output_shape=(self.action_size,))(state_value)

        action_advantage = Dense(512, activation="tanh", kernel_initializer='normal')(input_dueling)
        action_advantage = Dense(256, activation="tanh", kernel_initializer='normal')(action_advantage)
        action_advantage = Dense(128, activation="tanh", kernel_initializer='normal')(action_advantage)
        action_advantage = Dense(self.action_size, activation="linear", kernel_initializer='he_uniform')(
            action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - k.mean(a[:, :], keepdims=True),
                                  output_shape=(self.action_size,))(action_advantage)

        q_value = Add()([action_advantage, state_value])
        model = Model(input, q_value)
        model.summary()
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def _build_distance_estimator(self):

        if self.enable_dist_est_buffer:
            input_shape = self.state_size
        else:
            input_shape = list(self.state_size)
            input_shape[-1] = 1
            input_shape = tuple(input_shape)

        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3), padding="same", input_shape=self.state_size))
        model.add(Activation("relu"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, kernel_size=(3, 3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())
        # model.add(Dense(units=512, activation="tanh"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=1, activation="linear"))  # Estimates the distance
        model.summary()
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))

        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, distance):  # Add done here

        self.memory.append((state, action, reward, next_state, distance))  # and here

    def priority_error_calc(self, next_state, action, reward, q_values):
        # Must be called right after act function
        epsilon = 0.001
        self.next_state_buffer = self.buffer

        for i in range(self.buffer_size):
            if i != self.buffer_size - 1:
                self.next_state_buffer[:, :, :, :, i] = self.next_state_buffer[:, :, :, :, i + 1]  # shift the frames
            else:
                self.next_state_buffer[:, :, :, :, self.buffer_size - 1] = next_state  # add a new frame

        target_value = reward + self.gamma * \
                       self.target_model.predict(self.next_state_buffer)[0][
                           np.argmax(self.model.predict(self.next_state_buffer)[0])]
        error = np.abs(target_value - q_values[action]) + epsilon
        self.priority_values.append(error)
        self.replay_probabilities = self.probability_calculator()

    def probability_calculator(self):
        priority_scale_a = 0.7
        scaled_priorities = np.array(self.priority_values) ** priority_scale_a
        replay_probabilities = scaled_priorities / sum(scaled_priorities)
        return replay_probabilities

    def importance_weights(self, sample_priority_values):
        weights = ((1 / self.memory_size) * (1 / np.array(sample_priority_values))) ** (1 - self.coeff_b)
        normalized_weights = weights / max(weights)

        return normalized_weights

    def act(self, state):  # pass the distance value

        # The stack of 4 frames is used as a single state

        for i in range(self.buffer_size):
            if i != self.buffer_size - 1:
                self.buffer[:, :, :, :, i] = self.buffer[:, :, :, :, i + 1]  # shift the frames
            else:
                self.buffer[:, :, :, :, self.buffer_size - 1] = state  # add a new frame

        # if self.test_mode is False:

        act_values = self.model.predict(self.buffer)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), act_values[0]
        return np.argmax(act_values[0]), act_values[0]  # The second argument is the average of q values

    def estimate_distance(self):
        return self.distance_estimator.predict(self.buffer)[0]

    def flush_buffer(self):
        self.buffer = np.zeros((1, self.state_size[0], self.state_size[1], self.state_size[2], self.buffer_size))

    def replay(self, batch_size):

        # I exclude the first 3 frames from the memory.
        if self.enable_prioritized_replay:
            sample_indices = random.choices(range(len(self.memory)), k=batch_size, weights=self.replay_probabilities)
            sample_indices = np.array(sample_indices)

            enumerated_memory = list(enumerate(self.memory))
            # enumerated_memory = np.array(list(enumerate(self.memory)))
            # minibatch = enumerated_memory[sample_indices]
            minibatch = []
            for i in sample_indices:
                minibatch.append(enumerated_memory[i])

            # we have to obtain the probability of samples
            minibatch_probabilities = self.replay_probabilities[sample_indices]  # np array
            importance_weights = self.importance_weights(minibatch_probabilities)  # np array

            self.coeff_b = self.coeff_b * self.coeff_b_decay

        else:
            # minibatch = random.sample(
            # list(enumerate(deque(itertools.islice(self.memory, self.buffer_size - 1, self.memory_size)))), batch_size)
            minibatch = random.sample(list(enumerate(self.memory)), batch_size)

        update_input = np.zeros(
            (batch_size, self.state_size[0], self.state_size[1], self.state_size[2], self.state_size[3]))
        next_state_buffer = np.zeros(
            (batch_size, self.state_size[0], self.state_size[1], self.state_size[2], self.state_size[3]))
        update_target = np.zeros((batch_size, self.action_size))

        distance_target = np.zeros((batch_size, 1))

        # first batch for the state with 4 frames

        for i in range(batch_size):
            for j in range(self.buffer_size):
                index = minibatch[i][0] - self.buffer_size + j + 1
                if index < 0:
                    update_input[i, :, :, :, j] = np.zeros((self.state_size[0], self.state_size[1], self.state_size[2]))
                else:
                    update_input[i, :, :, :, j] = self.memory[index][0]

        # first batch for the next state with 4 frames
        for i in range(batch_size):
            for j in range(self.buffer_size):
                index = minibatch[i][0] - self.buffer_size + j + 1
                if index < 0:
                    next_state_buffer[i, :, :, :, j] = np.zeros(
                        (self.state_size[0], self.state_size[1], self.state_size[2]))
                else:
                    next_state_buffer[i, :, :, :, j] = self.memory[index][3]


        for i in range(batch_size):
            _state, action, reward, _next_state, distance = minibatch[i][1]

            target = self.model.predict(np.expand_dims(update_input[i], axis=0))[0]

            target[action] = reward + self.gamma * \
                             self.target_model.predict(np.expand_dims(next_state_buffer[i], axis=0))[0][
                                 np.argmax(self.model.predict(np.expand_dims(next_state_buffer[i], axis=0))[0])]

            update_target[i] = target
            # ******************DISTANCE ESTIMATION**********************

            distance_target[i] = distance

            # ***********************************************************
        if self.enable_prioritized_replay:

            self.model.fit(update_input, update_target, batch_size=batch_size, sample_weight=importance_weights,
                           epochs=1, verbose=0)
        else:
            self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

        # *******************UPDATE DISTANCE******************************
        if self.enable_dist_estimator:
            self.distance_estimator.fit(update_input, distance_target, batch_size=batch_size, epochs=1, verbose=0)

        # **************************************************************
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        self.target_model.save_weights(name)

    def load_dist_estimator(self, name):
        self.distance_estimator.load_weights(name)

    def save_dist_estimator(self, name):
        self.distance_estimator.save_weights(name)
