import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0


class TSPDataset(Dataset):
    def __init__(self, num_nodes, num_samples, random_seed=111):
        super(TSPDataset, self).__init__()
        torch.manual_seed(random_seed)

        self.data_set = []
        for l in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]


train_size = 100
val_size = 100

train_20_dataset = TSPDataset(10, train_size)
val_20_dataset = TSPDataset(10, val_size)
# num_inputs = 20
# num_actions = 10
# num_hidden = 128
#
# inputs = layers.Input(shape=(2,10))
# common = layers.Dense(num_hidden, activation="relu")(inputs)
# action = layers.Dense(num_actions, activation="softmax")(common)
# critic = layers.Dense(1)(common)
# model = keras.Model(inputs=inputs, outputs=[action, critic])

# Input layer
num_cities = 10
num_hidden = 128
inputs = layers.Input(shape=(2, num_cities))  # (x, y) coordinates of cities
# Flatten the input to be suitable for dense layers
flatten = layers.Flatten()(inputs)
# Hidden layer
hidden = layers.Dense(num_hidden, activation="relu")(flatten)
# Output layer for action probabilities
action_probs = layers.Dense(num_cities, activation="softmax")(hidden)
# Create the model
model = keras.Model(inputs=inputs, outputs=action_probs)

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0


# Create the TSP environment
class TSPEnvironment:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_nodes = dataset[0].size(1)
        self.current_episode = 0
        self.current_step = 0
        self.static = None
        self.mask = tf.ones(10)  # 1: den exei paei, 0 exei paei
        self.current_tour = []

    def calculate_distance(self, action):
        x1, y1 = self.static[:, self.current_step]
        x2, y2 = self.static[:, action]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def reset(self):
        self.current_episode += 1
        if self.current_episode >= len(self.dataset):
            self.current_episode = 0
        self.static = self.dataset[self.current_episode]
        self.current_step = 0
        self.mask = torch.ones(self.num_nodes, dtype=torch.float32)  # tf.Variable(tf.ones(self.num_nodes, tf.int32))
        # Initialize the state with the agent's starting city and remaining cities
        remaining_cities = torch.ones(self.num_nodes, dtype=torch.bool)
        remaining_cities[0] = 10e-1  # Mark the starting city as visited, false=visited
        self.state = self.static  # torch.cat((self.current_tour, remaining_cities), dim=-1)
        # self.state = torch.cat((self.current_tour, remaining_cities.unsqueeze(0)), dim=0)

        # state includes: [current_tour]: 2,num_nodes
        return self.state

    def isDone(self):
        comparison = tf.equal(self.mask, 1)  # exoume unvisited nodes?

        # Check if any element in the tensor is True
        hasUnvisitedNodes = tf.reduce_any(comparison)

        isDone = ~hasUnvisitedNodes

        if isDone:
            print(f'Current tour when done is:{self.current_tour}')
        return isDone

    def step(self, action):
        self.current_tour.append(action)
        next_step = action
        reward = -self.calculate_distance(
            action)  # -torch.norm(self.current_tour[self.current_step] - self.current_tour[next_step])
        self.current_step = next_step
        self.mask[action] = 0  # we mark city as visited
        done = self.isDone()
        next_state = self.state if not done else None
        return next_state, reward, done, {}

    def chooseAction(self, action_probs):
        masked_probabilities = action_probs *  self.mask
        sampled_indices = tf.random.categorical([masked_probabilities], num_samples=1)
        sampled_indices = sampled_indices[0].numpy()

        print(f'When mask is {self.mask} chosen index {sampled_indices[0]}')
        return sampled_indices[0]
        # return np.random.choice(len(action_probs), p=np.squeeze(action_probs))


# Create the TSP environment using the dataset
tsp_env = TSPEnvironment(train_20_dataset)

losses_per_epoch = []
rewards_per_epoch = []

while True:  # Run until solved
    state = tsp_env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            # action_probs, critic_value = model(state)
            action_probs = model(state)
            # critic_value_history.append(critic_value[0, 0])
            action_probs = tf.squeeze(action_probs)
            # Sample action from action probability distribution
            action = tsp_env.chooseAction(action_probs)
            # action_probs_history.append(tf.math.log(action_probs[0, action]))
            action_probs_history.append(tf.math.log(action_probs[action]))
            # Apply the sampled action in our environment
            state, reward, done, _ = tsp_env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        # history = zip(action_probs_history, critic_value_history, returns)
        history = zip(action_probs_history, returns)
        actor_losses = []
        # critic_losses = []

        # for log_prob, value, ret in history:
        for log_prob, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret  # - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # # The critic must be updated so that it predicts a better estimate of
            # # the future rewards.
            # critic_losses.append(
            #     huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            # )

        # Backpropagation
        loss_value = sum(actor_losses)  # + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        losses_per_epoch.append(loss_value)
        rewards_per_epoch.append(np.mean(rewards_history))

        # Clear the loss and reward history
        action_probs_history.clear()
        # critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 1 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if episode_count == 20:
        break

plt.plot(rewards_per_epoch)
plt.savefig('rewards.png')
plt.clf()
plt.plot(losses_per_epoch)
plt.savefig('losses.png')
