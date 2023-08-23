# -*- coding: utf-8 -*-
"""Neural Combinatorial Optimization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MZv6WIH_t3Q9b7lwNVtqSVtAX9uEgC5V

<p>This tutorial presents <a href="">Neural Combinatorial Optimization with Reinforcement Learning</a>. Focusing on the traveling salesman problem (TSP) and train a recurrent neural network that, given a set of city coordinates, predicts a distribution over different city permutations. Using
negative tour length as the reward signal, the model optimize the parameters of the recurrent
neural network using a policy gradient method. </p><p>Despite the computational expense, without much engineering and
heuristic designing, Neural Combinatorial Optimization achieves close to optimal
results on 2D Euclidean graphs with up to 100 nodes.</p><p>
Previous attempts used supervised learning. Learning from examples in such a
way is undesirable for NP-hard problems because (1) the performance of the model is tied to the
quality of the supervised labels, (2) getting high-quality labeled data is expensive and may be infeasible
for new problem statements, (3) one cares more about finding a competitive solution more than
replicating the results of another algorithm. By contrast, Reinforcement Learning (RL) provides an appropriate paradigm for training
neural networks for combinatorial optimization, especially because these problems have relatively
simple reward mechanisms that could be even used at test time. </p>
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

USE_CUDA = False

# Commented out IPython magic to ensure Python compatibility.
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt

# %matplotlib inline

"""<h3>Generating dataset for TSP Task</h3>"""


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


train_size = 100000  # 0
val_size = 1000  # 0

train_20_dataset = TSPDataset(20, train_size)
val_20_dataset = TSPDataset(20, val_size)

train_50_dataset = TSPDataset(50, train_size)
val_50_dataset = TSPDataset(50, val_size)

"""<h3>Reward</h3>
<p>
Given an input graph, represented as a sequence of n cities in a two dimensional space, the task is finding a permutation of the points π, termed a tour, that visits each city once and has the minimum
total length. Tthe length of a tour is defined by a permutation π.
</p>
"""


def reward(sample_solution, USE_CUDA=False):
    """
    Args: sample_solution seq_len of [batch_size]
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))

    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n - 1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i + 1], dim=1)

    tour_len += torch.norm(sample_solution[n - 1] - sample_solution[0], dim=1)

    return tour_len


class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=USE_CUDA):
        super(Attention, self).__init__()

        self.use_tanh = use_tanh
        self.C = C
        self.name = name

        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)), 1. / math.sqrt(hidden_size))

    def forward(self, query, ref):
        """
        Args:
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """

        batch_size = ref.size(0)
        seq_len = ref.size(1)

        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref = self.W_ref(ref)  # [batch_size x hidden_size x seq_len]
            expanded_query = query.repeat(1, 1, seq_len)  # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)

        elif self.name == 'Dot':
            query = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2)  # [batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)

        else:
            raise NotImplementedError

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits
        return ref, logits


"""<h3>Graph Embedding</h3>
<p>
This is simple node embedding. For more advanced methods see <a href="https://arxiv.org/pdf/1705.02801.pdf">Graph Embedding Techniques, Applications, and Performance: A Survey</a>
</p>
"""


class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=USE_CUDA):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda

        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded


"""<h3>Pointer Network</h3>
<p><a href="https://arxiv.org/abs/1506.03134">Pointer Networks
</a></p>
<p>Check my tutorial - <a href="https://github.com/higgsfield/np-hard-deep-reinforcement-learning/blob/master/Intro%20to%20Pointer%20Network.ipynb">intro to Pointer Networks</a></p>
<p>The model solves the problem of variable size output dictionaries using a recently proposed mechanism of neural attention. It differs from the previous attention attempts in that, instead of using attention to blend hidden units of an encoder to a context vector at each decoder step, it uses attention as a pointer to select a member of the input sequence as the output.</p>
<img src="imgs/Снимок экрана 2017-12-26 в 4.30.58 ДП.png">
"""


class PointerNet(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 attention,
                 use_cuda=USE_CUDA):
        super(PointerNet, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.seq_len = seq_len
        self.use_cuda = use_cuda

        self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention, use_cuda=use_cuda)

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask_to_logits(self, logits, mask, idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size x 1 x sourceL]
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(2)
        assert seq_len == self.seq_len

        embedded = self.embedding(inputs)
        encoder_outputs, (hidden, context) = self.encoder(embedded)

        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte()

        idxs = None

        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):

            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))

            query = hidden.squeeze(0)

            for i in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs)
            probs = F.softmax(logits)

            idxs = probs.multinomial(1).squeeze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    print(seq_len)
                    print(' RESAMPLE!')
                    idxs = probs.multinomial(1).squeeze(1)
                    break
            decoder_input = embedded[[i for i in range(batch_size)], idxs.data, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)

        return prev_probs, prev_idxs


"""<h3>Optimization with policy gradients</h3>
<p>Model-free policy-based Reinforcement Learning to optimize the parameters of a pointer network using the well-known REINFORCE algorithm</p>
"""


class CombinatorialRL(nn.Module):
    def __init__(self,
                 embedding_size,
                 hidden_size,
                 seq_len,
                 n_glimpses,
                 tanh_exploration,
                 use_tanh,
                 reward,
                 attention,
                 use_cuda=USE_CUDA):
        super(CombinatorialRL, self).__init__()
        self.reward = reward
        self.use_cuda = use_cuda

        self.actor = PointerNet(
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            use_cuda)

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch_size = inputs.size(0)
        input_size = inputs.size(1)
        seq_len = inputs.size(2)

        probs, action_idxs = self.actor(inputs)

        actions = []
        inputs = inputs.transpose(1, 2)
        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])

        action_probs = []

        for prob, action_id in zip(probs, action_idxs):
            action_probs.append(prob[[x for x in range(batch_size)], action_id.data])

        R = self.reward(actions, self.use_cuda)

        return R, action_probs, actions, action_idxs


embedding_size = 128
hidden_size = 128
n_glimpses = 1
tanh_exploration = 10
use_tanh = True

beta = 0.9
max_grad_norm = 2.

"""<h2>Creating two models:</h2>
<h3>1. Pointer Network with Dot Attention - TSP 20</h3><h3> 2. Pointer Network with Bahdanau Attention - TSP 50</h3>
"""

tsp_20_model = CombinatorialRL(
    embedding_size,
    hidden_size,
    20,
    n_glimpses,
    tanh_exploration,
    use_tanh,
    reward,
    attention="Dot",
    use_cuda=USE_CUDA)

tsp_50_model = CombinatorialRL(
    embedding_size,
    hidden_size,
    50,
    n_glimpses,
    tanh_exploration,
    use_tanh,
    reward,
    attention="Bahdanau",
    use_cuda=False)

"""<h3>Simple Training Class</h3>"""


class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, batch_size=128, threshold=None, max_grad_norm=2.):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.actor_optim = optim.Adam(model.actor.parameters(), lr=1e-4)
        self.max_grad_norm = max_grad_norm

        self.train_tour = []
        self.val_tour = []

        self.train_loss = []

        self.epochs = 0

    def train_and_validate(self, n_epochs):
        critic_exp_mvg_avg = torch.zeros(1)
        if USE_CUDA:
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        for epoch in range(n_epochs):
            for batch_id, sample_batch in enumerate(self.train_loader):
                self.model.train()

                inputs = Variable(sample_batch)

                R, probs, actions, actions_idxs = self.model(inputs)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs:
                    logprob = torch.log(prob)
                    logprobs += logprob

                logprobs[logprobs < -1000] = 0.

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.actor.parameters(), float(self.max_grad_norm), norm_type=2)

                self.train_loss.append(actor_loss.detach().item())
                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().data.item())

                if batch_id % 10 == 0:
                    self.plotTrainAndValidationTourLength(self.epochs)

                if batch_id % 100 == 0:
                    self.model.eval()
                    for val_batch in self.val_loader:
                        inputs = Variable(val_batch)
                        R, probs, actions, actions_idxs = self.model(inputs)
                        self.val_tour.append(R.mean().data.item())
            if self.threshold and self.train_tour[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break

            self.epochs += 1

    def plotTrainLoss(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.ylabel("Loss")
        plt.title(
            'train loss: epoch %s loss %s' % (epoch, self.train_loss[-1] if len(self.train_loss) else 'collecting'))
        plt.plot(self.train_loss)
        plt.grid()

    def plotTrainAndValidationTourLength(self, epoch):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.ylabel("Tour length")
        plt.title('train tour length: epoch %s reward %s' % (
            epoch, self.train_tour[-1] if len(self.train_tour) else 'collecting'))
        plt.plot(self.train_tour)
        plt.grid()
        plt.subplot(132)
        plt.title(
            'val tour length: epoch %s reward %s' % (epoch, self.val_tour[-1] if len(self.val_tour) else 'collecting'))
        plt.plot(self.val_tour)
        plt.grid()
        plt.show()


"""<h2>TSP 20 Results of Pointer Network with Dot Attention</h2>"""
if __name__ == '__main__':
    tsp_20_train = TrainModel(tsp_20_model,
                              train_20_dataset,
                              val_20_dataset,
                              threshold=3.99)
    tsp_20_train.train_and_validate(5)
    """<h2>TSP 50 Results of Pointer Network with Bahdanau Attention</h2>"""

    train_50_train = TrainModel(tsp_50_model,
                                train_50_dataset,
                            val_50_dataset,
                            threshold=6.4)


    train_50_train.train_and_validate(10)