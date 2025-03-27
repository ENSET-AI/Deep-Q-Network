from Utils.Parameters import *
from Utils.GridWorld import *

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from collections import deque
import random


class DDQNAgent:
    def __init__(self):
        """Agent DQN use neural network to train"""
        self.state_size = STATE_SIZE
        self.action_size = ACTION_SIZE
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        """Build a neural network

        Returns:
            _type_: Model with 2 hidden layers each 24 neurons
        """
        model = Sequential(
            [
                Input(shape=(self.state_size,)),
                Dense(
                    24,
                    activation="relu",
                ),
                Dense(
                    24,
                    activation="relu",
                ),
                Dense(self.action_size, activation="linear"),
            ]
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

        return model

    def update_target_model(self):
        """Copy weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Choose an action based on ε-greedy strategy."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return np.argmax(q_values)

    def replay(self):
        """Train the model with past experiences."""
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in batch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + GAMMA * np.max(
                    self.target_model.predict(np.array([next_state]), verbose=0)[0]
                )

            self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)

        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def train(self, episodes: int, grid_world: GridWorld, period=10):
        """Train the agent using the DDQN algorithm."""
        for ep in range(episodes):
            state = grid_world.reset()
            total_reward = 0

            for step in range(50):
                action = self.act(state)
                next_state, reward, done = grid_world.step(action)

                self.remember(
                    action=action,
                    state=state,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                )
                state = next_state
                total_reward += reward
                if done:
                    break

            self.replay()
            if ep % period == 0:
                self.update_target_model()

            print(f"Episode {ep+1}/{episodes} :")
            print(f"\tScore: {total_reward}")
            print(f"\tEpsilon: {self.epsilon:.2f}")
