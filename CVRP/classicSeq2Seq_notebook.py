import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Implementation: https://github.com/ajayn1997/RL-VRP-PtrNtwrk/tree/main
from train_utilities.EarlyStopping import EarlyStopping

train_size = 100  # 00#000  # 0#000  # 1000000
batch_size = 8  # 256
validation_size = 100  # 0#0
hidden_size = 128
lr = 5e-4  # 0.0005
epochs = 3  # 0
seq_len = 20


# train_size = 500000  # 0#000  # 1000000
# batch_size = 256
# validation_size = 500  # 0#0
# hidden_size = 128
# lr = 5e-4  # 0.0005
# epochs = 35
# seq_len = 20


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size), requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size), requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        batch_size, hidden_size, seq_len = static_hidden.size()
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        context = rnn_out.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, dynamic_hidden, context), dim=1)
        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)  # [batch_size, seq_len]

        assert probs.size(0) == batch_size
        assert probs.size(1) == seq_len
        return probs, last_hh


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Seq2SeqModelCVRP(nn.Module):
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

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(Seq2SeqModelCVRP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder & decoder models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)

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
                logp = m.log_prob(ptr)  # log probability pithanothta epiloghs action ptr given input m
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
                logp = logp * (1. - is_done)  # logp: pithanothta epiloghs action y_1 given input x_1

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic, ptr.data).detach()

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2, ptr.view(-1, 1, 1).expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Detected device {}'.format(device))
from torch.nn.functional import normalize


def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.', num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    losses = []

    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, tour_logp = actor.forward(static, dynamic, x0)

            distanceTravelled = reward_fn(static, tour_indices).mean().item()
            advantage = - distanceTravelled

            loss = torch.mean(advantage * tour_logp.sum(dim=1))  # minimize to -advantage aka max(advantage)

            rewards.append(np.mean(advantage))
            losses.append(torch.mean(loss.detach()).item())

        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png' % (batch_idx, advantage)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards), np.mean(losses)


def train(actor, task, num_nodes, train_data, valid_data, reward_fn,
          render_fn, batch_size,
          epochs,
          experiment_details,
          actor_lr, critic_lr,
          max_grad_norm, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)

    print('Starting training')

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_reward = np.inf

    loss_at_epoch = []
    reward_at_epoch = []
    average_advantage_validation_data = []
    average_loss_validation_data = []
    training_time_per_epoch = []

    early_stopping = EarlyStopping(patience=7, verbose=True, delta=200)
    start_time = time.time()

    for epoch in range(epochs):

        actor.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through the dataset
            tour_indices, tour_logp = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            distanceTravelled = reward_fn(static, tour_indices)
            advantage = -distanceTravelled
            loss = torch.mean(advantage * tour_logp.sum(dim=1))

            actor_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            rewards.append(torch.mean(advantage.detach()).item())
            losses.append(torch.mean(loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' % (
                    batch_idx, len(train_loader), mean_reward, mean_loss, times[-1]))

        # Epoch finished
        training_time_per_epoch.append(np.sum(times))
        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'seq2seqCVRP.pt')

        torch.save(actor.state_dict(), save_path)

        # Save rendering of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid_reward, mean_valid_loss = validate(valid_loader, actor, reward_fn, render_fn, valid_dir, num_plot=5)
        early_stopping(mean_valid_reward, actor, path="checkpoints\\modelCVRPAttention")
        if early_stopping.early_stop:
            print(f"-------------------Early stopping at epoch {epoch}-------------------")
            break

        # Save best model parameters
        if mean_valid_reward < best_reward:
            best_reward = mean_valid_reward
            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

        average_advantage_validation_data.append(mean_valid_reward)
        average_loss_validation_data.append(mean_valid_loss)
        print('Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid_reward, time.time() - epoch_start, np.mean(times)))

        loss_at_epoch.append(mean_loss)
        reward_at_epoch.append(mean_reward)

    # Training finished
    modelName = "classicSeqToSeqCVRP"
    training_time_in_secs = time.time() - start_time
    print("--- %s seconds ---" % (training_time_in_secs))
    f = open("training_metricsclassicSeqToSeqCVRP.txt", "a")
    f.write("\n\n ------------------- New experiment -------------------- \n\n")

    f.write(f"Experiment name: {experiment_details}\n")
    f.write(f"Training for {training_time_in_secs} seconds")
    f.write(f"Stopped at epoch {epoch} out of {epochs}\n")
    f.write(f"Experiment Time: {get_filename_time()}\n")

    # metrics_folder = f"{modelName}\\nodes{num_nodes}"#\\train{train_size}"
    plot_train_loss_and_train_reward(epoch,
                                     loss_at_epoch,
                                     reward_at_epoch,
                                     experiment_details,
                                     modelName)
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


def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


def train_vrp(args):
    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    print('Starting VRP training')

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    print('Train data: {}'.format(train_data))
    valid_data = VehicleRoutingDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)

    actor = Seq2SeqModelCVRP(STATIC_SIZE,
                             DYNAMIC_SIZE,
                             args.hidden_size,
                             train_data.update_dynamic,
                             train_data.update_mask,
                             args.num_layers,
                             args.dropout).to(device)
    print('Actor: {} '.format(actor))

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = reward
    kwargs['render_fn'] = render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, reward, render, test_dir, num_plot=5)

    print('Average tour length: ', out)


if __name__ == '__main__':
    experiment_details = f'bmmseq2seqCVRP_epoch{epochs}_trainSize{train_size}_bs{batch_size}_valSize{validation_size}_hidden{hidden_size}_lr{lr}'
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')

    parser.add_argument('--experiment_details', default=experiment_details)
    parser.add_argument('--epochs', default=epochs, type=int)
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=seq_len, type=int)
    parser.add_argument('--actor_lr', default=lr, type=float)
    parser.add_argument('--critic_lr', default=lr, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=batch_size, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=hidden_size, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=train_size, type=int)
    parser.add_argument('--valid-size', default=validation_size, type=int)

    args = parser.parse_args()

    # print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    # print(args.checkpoint)

    train_vrp(args)
