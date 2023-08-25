# -*- coding: utf-8 -*-
"""worked-cvrp-classic-seq2seq.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qVP9Bx9msR6sNiUcP1fPENJkVgmYdgR6
"""
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import datetime
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import torch

"""Αρχικά ορίζουμε τα δεδομένα που αποτελούν το περιβάλλον του προβλήματος. Στο πρόβλημα ορίζεται και μια συνάρτηση ανανέωσης του περιβάλλοντος η οποία ανανεώνει τα δυναμικά χαρακτηριστικά του προβλήματος."""

"""Defines the main task for the VRP.
The VRP is defined by the following traits:
    1. Each city has a demand in [1, 9], which must be serviced by the vehicle
    2. Each vehicle has a capacity (depends on problem), the must visit all cities
    3. When the vehicle load is 0, it __must__ return to the depot to refill
"""

from torch.utils.data import Dataset
import matplotlib

matplotlib.use('Agg')


class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, max_load=20, max_demand=9,
                 seed=None):
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        # Depot location will be the first node in each
        locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = locations

        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        # All states will have their own intrinsic demand in [1, max_demand),
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30,
        # demands will be scaled to the range (0, 3)
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)

        demands[:, 0, 0] = 0  # depot starts with a demand of 0
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])

    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.

        Parameters
        ----------
        dynamic: torch.autograd.Variable of size (1, num_feats, seq_len)
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads)

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        # if (1 - repeat_home).any():
        if (~repeat_home).any():
            # new_mask[(1 - repeat_home).nonzero(), 0] = 0.
            new_mask[(~repeat_home).nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():
            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.tensor(tensor.data)


"""Στη συνέχεια θα εκπαιδεύσουμε το μοντέλο

Στην συνέχεια ορίζουμε τις συναρτήσεις που χρησιμοποιούμε για να δούμε κάποιες μετρικές του μοντέλου.
"""


def plot_train_and_validation_loss(epoch, train_loss, val_loss, experiment_details="", folder_name=None):
    title1 = f"Train loss epoch {epoch}, {train_loss[-1]:.2f}"
    title2 = f"Val loss epoch {epoch},{val_loss[-1]:.2f}"
    filename = f"losses_{experiment_details}_date{get_filename_time()}.png"

    os.makedirs(folder_name, exist_ok=True)
    plot_train_and_validation_metrics("Loss", train_loss, val_loss, title1, title2, filename, folder_name)


def plot_train_loss_and_train_reward(epoch, train_loss, train_reward, experiment_details="", folder_name=None):
    title1 = f"Train loss epoch {epoch}, {train_loss[-1]:.2f}"
    title2 = f"Train reward epoch {epoch},{train_reward[-1]:.2f}"
    filename = f"train_metrics_{experiment_details}_date{get_filename_time()}.png"

    os.makedirs(folder_name, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    axes[0].plot(train_loss)
    axes[0].set_title(title1)
    axes[0].set(xlabel='Epochs', ylabel="Loss")
    axes[0].grid()
    axes[1].grid()
    axes[1].plot(train_reward)
    axes[1].set(xlabel='Epochs', ylabel="Reward")
    axes[1].set_title(title2)

    now = datetime.datetime.now()
    directory = f'metrics\\day_{now.day}\\hour_{now.hour}'

    if folder_name is not None:
        directory = f'metrics\\day_{now.day}\\hour_{now.hour}\\{folder_name}'

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\{filename}')

    plt.clf()


def get_filename_time():
    now = datetime.datetime.now()
    return f'm={now.month}_d={now.day}_h={now.hour}_m={now.minute}'


def plot_train_and_validation_distance(epoch, train_reward, val_reward, experiment_details="", folder_name=None):
    title1 = f"Mean distance, train data, epoch {epoch}"
    title2 = f"Mean distance, validation data, epoch {epoch}"
    filename = f"distances_{experiment_details}_date{get_filename_time()}.png"
    os.makedirs(folder_name, exist_ok=True)
    plot_train_and_validation_metrics("Distance", train_reward, val_reward, title1, title2, filename, folder_name)


def plot_train_and_validation_reward(epoch, train_reward, val_reward, experiment_details="", folder_name=None):
    title1 = f"Train reward epoch {epoch}"
    title2 = f"Val reward epoch {epoch}"
    filename = f"rewards_{experiment_details}_date{get_filename_time()}.png"
    os.makedirs(folder_name, exist_ok=True)
    plot_train_and_validation_metrics("Reward", train_reward, val_reward, title1, title2, filename, folder_name)


def plot_critic_estimate_and_loss(criticLoss, criticEstimate, experiment_details, folder_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    filename = f"CRITIC_{experiment_details}_date{get_filename_time()}.png"

    ax = axes.flat
    ax[0].set(xlabel='Epochs', ylabel="Loss")
    ax[1].set(xlabel='Epochs', ylabel="Estimate")

    axes[0].plot(criticLoss)
    axes[0].set_title("Critic Loss")

    axes[0].grid()
    axes[1].grid()
    axes[1].plot(criticEstimate)

    axes[1].set_title("Critic Estimate")

    now = datetime.datetime.now()
    directory = f'metrics\\day_{now.day}\\hour_{now.hour}'
    if folder_name is not None:
        directory = f'metrics\\day_{now.day}\\hour_{now.hour}\\{folder_name}'
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\{filename}')

    plt.clf()


def plot_train_and_validation_metrics(metric_name, train_metric, val_metric,
                                      title1, title2,
                                      filename,
                                      folder_name=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Add x and y labels to all subplots
    for ax in axes.flat:
        ax.set(xlabel='Epochs', ylabel=f'{metric_name}')

    axes[0].plot(train_metric)
    axes[0].set_title(title1)
    axes[0].grid()
    axes[1].grid()
    axes[1].plot(val_metric)
    axes[1].set_title(title2)

    # directory =f'metrics\\{epoch}'
    now = datetime.datetime.now()
    directory = f'metrics\\day_{now.day}\\hour_{now.hour}'

    if folder_name is not None:
        directory = f'metrics\\day_{now.day}\\hour_{now.hour}\\{folder_name}'

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\{filename}')

    plt.clf()


"""Στη συνέχεια ορίζουμε το μοντέλο μας"""


class EarlyStopping:
    def __init__(self, patience=30, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        os.makedirs(path, exist_ok=True)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:  # to score mas einai mikrotero apo to kalutero mas
            # ara mallon tha xreiastei na stamathsoume
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # brikame kalutero scor! Wohoo
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):

        ## TODO: add custom name for model!!! We want to reload them later
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


"""Τέλος εκπαιδεύουμε το μοντέλο μας."""


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    losses = []
    distances = []

    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        x0 = x0 if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, tour_logp = actor.forward(static, dynamic, x0)

            distanceTravelled = reward_fn(static, tour_indices)

            critic_est = critic(static, dynamic).view(-1)
            print(f'Distance travelled:{distanceTravelled} and mean {distanceTravelled.mean()} ')
            distanceTravelled = distanceTravelled - distanceTravelled.mean()
            # TRY THIS ONE TOO
            #distanceTravelled = distanceTravelled / (distanceTravelled.std() + 1e-8)
            print(f'Critic estimate: {critic_est.detach()}')
            print(f'Distance travelled now :{distanceTravelled.detach()} ')
            advantage = -distanceTravelled - critic_est

            # TRY normalizing the advantages
            #advantages = advantages - advantages.mean()
            #advantages = advantages / (advantages.std() + 1e-8)  # The small term ensures we don't divide by zero.

            ## TODO: this is new change try this
            # TODO 2: try the advantage = -distance-critic but with advantage.detach()!!
            # TODO 3: try distance = distance - distance.mean() kai advantage = -distance
            loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))

            print(f'loss is {loss.detach()}')
            print(f'Advantage is {advantage.detach()}')

            rewards.append(torch.mean(advantage))
            losses.append(torch.mean(loss.detach()).item())
            distances.append(torch.mean(distanceTravelled))

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png' % (batch_idx, -distanceTravelled)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards), np.mean(losses), np.mean(distances)


def train(actor, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size, validation_batch_size,
          epochs,
          experiment_details,
          actor_lr, critic_lr,
          max_grad_norm, modelName=None, task="vrp", verbose=True):
    """Performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, '../checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(actor_optim, 'min')
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, validation_batch_size, False, num_workers=0)

    best_reward = np.inf

    loss_at_epoch = []
    critic_estimate_per_epoch = []
    critic_losses_per_epoch = []
    reward_at_epoch = []
    distances_at_epoch = []
    average_advantage_validation_data = []
    average_distance_validation_data = []
    average_loss_validation_data = []

    early_stopping = EarlyStopping(patience=20, verbose=verbose, delta=200)  # TODO: check how big should delta be
    start_time = time.time()

    for epoch in range(epochs):

        actor.train()

        times, losses, rewards, critic_losses_at_epoch, critic_estimates, distances = [], [], [], [], [], []

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            static, dynamic, x0 = batch
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            distanceTravelled = reward_fn(static, tour_indices)

            critic_est = critic(static, dynamic).view(-1)
            distanceTravelled = distanceTravelled - distanceTravelled.mean()

            advantage = - distanceTravelled - critic_est# we want to maximize the advantage
            loss = torch.mean(advantage.detach() * tour_logp.sum(dim=1))

            actor_optim.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_loss = torch.mean(advantage ** 2)
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_estimates.append(torch.mean(critic_est.detach()).item())
            critic_losses_at_epoch.append(torch.mean(critic_loss.detach()).item())
            rewards.append(torch.mean(advantage.detach()).item())
            losses.append(torch.mean(loss.detach()).item())
            distances.append(torch.mean(distanceTravelled.detach()).item())

        # Epoch finished
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)
        mean_distance = np.mean(distances)
        mean_critic_loss = np.mean(critic_losses_at_epoch)
        mean_critic_estimate = np.mean(critic_estimates)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, f'seq2seqCVRP{experiment_details}.pt')

        torch.save(actor.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid_reward, mean_valid_loss, mean_validation_distance = validate(valid_loader, actor, reward_fn,
                                                                                render_fn, valid_dir, num_plot=5)
        early_stopping(mean_valid_reward, actor, path="../checkpoints/modelCVRPAttention")
        if early_stopping.early_stop:
            print(f"-------------------Early stopping at epoch {epoch}-------------------")
            break

        # Save best model parameters
        if mean_valid_reward < best_reward:
            best_reward = mean_valid_reward
            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

        average_advantage_validation_data.append(mean_valid_reward)
        average_distance_validation_data.append(mean_validation_distance)
        average_loss_validation_data.append(mean_valid_loss)

        if verbose:
            print(f"Mean epoch loss: {mean_loss}, reward:{mean_reward}, val reward:{mean_valid_reward}")

        loss_at_epoch.append(mean_loss)
        reward_at_epoch.append(mean_reward)
        distances_at_epoch.append(mean_distance)
        critic_losses_per_epoch.append(mean_critic_loss)
        critic_estimate_per_epoch.append(mean_critic_estimate)
        scheduler.step(mean_valid_loss)

        if verbose:
            print('Epoch-{0} lr: {1}'.format(epoch, actor_optim.param_groups[0]['lr']))

    # Training finished
    if modelName is None:
        modelName = "mean"

    training_time_in_secs = time.time() - start_time
    print("--- %s seconds ---" % (training_time_in_secs))

    f = open("../training_metricsACTORCRITIC_FINAL.txt", "a")
    f.write("\n\n ------------------- New experiment -------------------- \n\n")
    f.write(f"Experiment name: {experiment_details}\n")
    f.write(f"Training for {training_time_in_secs} seconds aka {training_time_in_secs // 60} minutes")
    f.write(f"Stopped at epoch {epoch} out of {epochs}\n")
    f.write(f"Experiment Time: {get_filename_time()}\n")

    # metrics_folder = f"{modelName}\\nodes{num_nodes}"#\\train{train_size}"
    plot_train_and_validation_distance(epoch,
                                       distances_at_epoch,
                                       average_distance_validation_data,
                                       experiment_details,
                                       folder_name=modelName)

    plot_critic_estimate_and_loss(critic_losses_per_epoch, critic_estimate_per_epoch, experiment_details,
                                  folder_name=modelName)
    plot_train_and_validation_loss(epoch,
                                   loss_at_epoch,
                                   average_loss_validation_data,
                                   experiment_details,
                                   modelName)

    plot_train_and_validation_reward(epoch,
                                     reward_at_epoch,
                                     average_advantage_validation_data,
                                     experiment_details,
                                     folder_name=modelName)


"""Ορίζουμε τη συνάρτηση ανταμοιβής"""


def reward(static, tour_indices):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1)


"""Στη συνέχεια δοκιμάζουμε να εκπαιδεύσουμε το μοντέλο προσθέτοντας μηχανισμό προσοχής."""


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        '''
        static_hidden: [batch_size, hidden_size, sequence_length +1]
        dynamic_hidden: [batch_size, hidden_size, sequence_length +1]
        decoder_hidden: [batch_size, 1, hidden_size]
        '''
        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.transpose(2, 1).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)

        return attns


class PointerAttention(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(PointerAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size), requires_grad=True))

        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)  # [batch_size, seq_len]

        return probs, last_hh


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # TODO: add activation function? add tsp encoder?

    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)

        return output  # (batch, hidden_size, seq_len)


class Seq2SeqModelCVRPWithAttention(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidlines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size, update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(Seq2SeqModelCVRPWithAttention, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = PointerAttention(hidden_size, num_layers, dropout)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad=True)

    def forward(self, static, dynamic, decoder_input=None, last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """

        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device=device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # Static elements only need to be processed once, and can be used across
        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        for _ in range(max_steps):
            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(static_hidden, dynamic_hidden, decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)

                ptr = m.sample()
                while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                    ptr = m.sample()
                logp = m.log_prob(ptr)
            else:
                prob, ptr = torch.max(probs, 1)  # Greedy
                logp = prob.log()

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(dynamic, ptr.data)
                dynamic_hidden = self.dynamic_encoder(dynamic)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2, ptr.view(-1, 1, 1).expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StateCritic(nn.Module):
    """Estimates the problem complexity.
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output



render = None

if __name__ == '__main__':
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)
    weight_decay = 1e-5  # san na ta xeiroterepse to weight decay...
    validation_bs = 5
    validation_size = 100
    hidden_size = 256
    epochs = 20
    seq_len = 10

    seed = 12345
    test = False
    checkpoint = None
    task = "vrp"
    nodes = seq_len
    actor_lr = 5e-5
    critic_lr = 5e-6  # 0.0005

    hidden = hidden_size
    step = 5
    layers = 1
    gamma = 0.1
    max_grad_norm = 4.
    dropout = 0.2

    max_load = LOAD_DICT[seq_len]
    train_size = 15000  # _000
    train_data = VehicleRoutingDataset(train_size,
                                       seq_len,
                                       max_load,
                                       MAX_DEMAND,
                                       seed)

    valid_data = VehicleRoutingDataset(validation_size,
                                       seq_len,
                                       max_load,
                                       MAX_DEMAND,
                                       seed + 1)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, hidden_size).to(device)


    model = Seq2SeqModelCVRPWithAttention(STATIC_SIZE,
                                          DYNAMIC_SIZE,
                                          hidden_size,
                                          train_data.update_dynamic,
                                          train_data.update_mask,
                                          num_layers=1,
                                          dropout=dropout)


    bss = [128, 256, 512]
    for i in range(len(bss)):
        bs = bss[i]
        experiment_details = f'actorCriticCVRP_epoch{epochs}_trainSize{train_size}_bs{bs}_valSize{validation_size}_hidden{hidden_size}'

        train(model, seq_len,
              train_data, valid_data,
              reward,
              render,
              bs,
              validation_bs,
              epochs,
              experiment_details,
              actor_lr,
              critic_lr,
              max_grad_norm, task="vrp")
