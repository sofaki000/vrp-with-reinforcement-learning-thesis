import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np

# discount factor for future utilities
from torch.utils.data import Dataset
DISCOUNT_FACTOR = 0.99
# number of episodes to run
NUM_EPISODES = 100
# max steps per episode
MAX_STEPS = 100#00
# device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TSPDataset(Dataset):
    """  Random TSP dataset """

    def __init__(self, data_size, seq_len):
        self.data_size = data_size
        self.seq_len = seq_len
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        sample = {'Points': tensor}
        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i + 1, self.data_size))
            points_list.append(np.random.random((self.seq_len, 2)))
        return {'Points_List': points_list}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec
class Environment():
    def __init__(self):
        super();
        self.dataset = TSPDataset(100, 5).data['Points_List']
    def get_reward(self, tour_indices):
        '''static: [batch_size, 2, sequence_length]
            tour_indices: [batch_size, tour_length]
            Euclidean distance between all cities / nodes given by tour_indices
        '''

        dataset = torch.tensor(self.dataset).transpose(1,2)
        # Convert the indices back into a tour
        idx = tour_indices.unsqueeze(1).expand(-1, dataset .size(1), -1)
        tour = torch.gather(dataset.data, 2, idx).permute(0, 2, 1)
        # Euclidean distance between each consecutive point
        tour_len = torch.sqrt(
                torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))  # [batch_size, sequence_length]

        return tour_len.sum(1)

    def getState(self):
        # returns
        return self.dataset
    def step(self, action):
        reward = self.get_reward(action)
        #returns new_state, reward, done, _, _
        return self.dataset, reward, False


# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):
    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, action_space)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x.float())

        # relu activation
        x = F.relu(x)

        # actions
        actions = self.output_layer(x)

        # get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=1)

        return action_probs


def select_action(network, state):
    ''' Selects an action given current state
    Args:
    - network (Torch NN): network to process state
    - state (Array): Array of action space in an environment

    Return:
    - (int): action that is selected
    - (float): log probability of selecting that action given state and network
    '''
    # convert state to float tensor, add 1 dimension, allocate tensor on device
    #state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    state = torch.from_numpy(np.array(state)).to(DEVICE)
    # use network to predict action probabilities
    action_probs = network(state)

    # sample an action using the probability distribution
    m = Categorical(action_probs)
    action = m.sample()

    # return action
    # gyrnaei MIA EPILOGH: action:[batch_size, 1]
    return action , m.log_prob(action)


# Make environment
env =  Environment()
observation_space, action_space = 2, 5
# Init network
network = PolicyNetwork(observation_space,action_space).to(DEVICE)

# Init optimizer
optimizer = optim.Adam(network.parameters(), lr=1e-2)

state = env.getState()#to state einai oi syntetagmenes
select_action(network, state)
# track scores
scores = []

# iterate through episodes
for episode in (range(NUM_EPISODES)):

    # reset environment, initiable variables
    state = env.getState()

    rewards = []
    log_probs = []
    score = 0

    # generate episode
    for step in range(MAX_STEPS):

        # select action
        action, lp = select_action(network, state)

        # execute action
        new_state, reward, done = env.step(action)

        # track episode score
        score += reward

        # store reward and log probability
        rewards.append(reward)
        log_probs.append(lp)

        # end episode
        if done:
            break

        # move into new state
        state = new_state

    # append score
    scores.append(score)

    # Calculate Gt (cumulative discounted rewards)
    discounted_rewards = []

    # track cumulative reward
    total_r = 0

    # iterate rewards from Gt to G0
    for r in reversed(rewards):
        # Base case: G(T) = r(T)
        # Recursive: G(t) = r(t) + G(t+1)^DISCOUNT
        total_r = r + total_r ** DISCOUNT_FACTOR
        # append to discounted rewards
        discounted_rewards.append(total_r)

    # reverse discounted rewards
    rewards = torch.tensor(discounted_rewards).to(DEVICE)
    rewards = torch.flip(rewards, [0])

    # adjusting policy parameters with gradient ascent
    loss = []
    for r, lp in zip(rewards, log_probs):
        # we add a negative sign since network will perform gradient descent and we are doing gradient ascent with REINFORCE
        loss.append(-r * lp)

    # Backpropagation
    optimizer.zero_grad()
    sum(loss).backward()
    optimizer.step()



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('Training score of CartPole with REINFORCE')

