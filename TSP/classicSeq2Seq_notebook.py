import time
import numpy as np
from train_utilities.EarlyStopping import EarlyStopping
import torch.nn.functional as F
import torch
from torch import nn
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from TSP.tsp_dataset import TSPDataset
from plotting_metrics.format_tour import format_tour
from plotting_metrics.plot_loss_reward import plot_train_and_validation_loss, plot_train_and_validation_reward, \
    get_filename_time

static_features = 2
hidden_size = 128


# experiments: arxika stohastic kai deterministic me argmax thing
# twra tha dokimasw na mhn dinw ta targets ston decoder ws input, alla to prohgoumeno input
# pou tha dinetai mesw enos embedding ofc

## TODO: fix bug that outputs the same route all the time FOR SMALL NUMBER OF NODES!!!


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        dropout_p = 0.1
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(static_features, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, sequence_length):
        super(Decoder, self).__init__()

        self.embedding = nn.Linear(1, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.out = nn.Linear(hidden_size, sequence_length)

    def apply_mask_to_logits(self, logits, mask, indexes):
        batch_size = logits.size(0)
        clone_mask = mask.clone()
        if indexes is not None:
            clone_mask[[i for i in range(batch_size)], indexes.data.squeeze(1).long()] = 1

            logits[clone_mask.unsqueeze(1)] = -np.inf
        else:
            logits[:, :] = -np.inf
            # we want to start from depot, ie the first node
            logits[:, :, 0] = 1

        return logits, clone_mask

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        decoder_input = torch.ones(batch_size, 1)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        tours = []
        tour_logp = []
        mask = torch.zeros(batch_size, seq_len).byte()

        chosen_indexes = None
        for i in range(seq_len):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1).float()  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)  # todo: understand this topi gia na to baleis ws exhghsh
                decoder_input = topi.squeeze(-1).detach().float()  # detach from history as input

            masked_logits, mask = self.apply_mask_to_logits(decoder_output, mask, chosen_indexes)

            # We transform decoder output to the actual result
            chosen_indexes = torch.argmax(masked_logits, dim=2).float()  # [batch_size, 1]
            log_probs = F.log_softmax(decoder_output, dim=2)
            logp = torch.gather(log_probs, 2, chosen_indexes.unsqueeze(2).long()).squeeze(2)  # [batch_size, 1]

            tour_logp.append(logp)  # logp: to log probability na parei to action pou kaname sample,
            tours.append(chosen_indexes.unsqueeze(1))

        tours = torch.cat(tours, 2)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tours, tour_logp

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output.unsqueeze(1), hidden)
        output = self.out(output)
        return output, hidden


def reward_fn(static, tour_indices):
    """
    static: [batch_size, 2, sequence_length]
    tour_indices: [batch_size, tour_length]
    Euclidean distance between all cities / nodes given by tour_indices
    """
    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)
    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))  # [batch_size, sequence_length]

    return tour_len.sum(1)


class ClassicSeq2SeqTSPModel(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(sequence_length)

    def forward(self, inputs, targets=None):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        tours, tour_logp = self.decoder(encoder_outputs, encoder_hidden, targets)
        return tours, tour_logp


def trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
                                                    test_dataset,
                                                    epochs,
                                                    experiment_details,
                                                    batch_size=10,
                                                    validation_batch_size=20,
                                                    num_nodes=13,
                                                    lr=1e-4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=1)

    model = ClassicSeq2SeqTSPModel(sequence_length=num_nodes)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_loss = []
    train_loss = []

    val_advantages_per_epoch = []
    advantages_per_epoch = []

    val_losses_per_epoch = []
    losses_per_epoch = []

    tour_lengths_per_epoch = []

    # Training with early stopping
    early_stopping = EarlyStopping(patience=7, verbose=True, delta=1000)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()

        loss_at_epoch = 0.0
        val_loss_at_epoch = 0.0
        advantage_at_epoch = 0.0
        val_advantage_at_epoch = 0.0

        iterator = tqdm(train_loader, unit='Batch')
        average_tour_length_per_epoch = []

        for batch_id, sample_batch in enumerate(iterator):
            optimizer.zero_grad()

            train_batch = Variable(sample_batch['Points'])

            output_routes, tour_logp = model(train_batch)

            reward = reward_fn(train_batch.transpose(1, 2), output_routes.squeeze(1).to(torch.int64))
            advantage = -reward.mean()

            loss = torch.mean(advantage * tour_logp.sum(1))

            loss.backward()  # computes the gradients of the loss function with respect to the weights of the network
            optimizer.step()  # updates the weights to minimize the loss

            loss_at_epoch += loss.detach().sum().item()
            advantage_at_epoch += torch.sum(advantage.detach())

            distances_per_batch = reward_fn(sample_batch['Points'].transpose(1, 2),
                                            output_routes.squeeze(1).to(torch.int64))
            average_tour_length_per_epoch.append(distances_per_batch.sum(0).item())

        tour_lengths_per_epoch.append(np.mean(average_tour_length_per_epoch))

        # epoch finished
        with torch.no_grad():
            model.eval()
            for val_batch in validation_loader:
                train_batch = Variable(val_batch['Points'])

                tours, tour_logp = model(train_batch)
                reward = reward_fn(train_batch.transpose(1, 2), tours.squeeze(1).to(torch.int64))

                val_advantage = -reward.mean()

                # we want to maximize the "loss", which is the negative tour length. The minus is inside the tour_logp
                loss = torch.mean(val_advantage * (tour_logp.sum(1)))
                val_loss.append(loss.data.item())
                val_loss_at_epoch += loss.detach().item()
                val_advantage_at_epoch += torch.sum(val_advantage.detach())
                # for this tour, what did the model find and what did the target was?
                format_tour(tours.squeeze(1).int())

        ### Epoch finished
        early_stopping(val_loss_at_epoch, model, path="checkpoints\\model")
        if early_stopping.early_stop:
            print(f"-------------------Early stopping at epoch {epoch}-------------------")
            break

        train_loss.append(loss_at_epoch / batch_size)

        print(f'Loss at epoch {epoch}: {loss_at_epoch}')
        losses_per_epoch.append(loss_at_epoch)
        val_losses_per_epoch.append(val_loss_at_epoch)

        advantages_per_epoch.append(advantage_at_epoch)
        val_advantages_per_epoch.append(val_advantage_at_epoch)

    # Training finished
    training_time_in_secs = time.time() - start_time
    print("--- %s seconds ---" % (training_time_in_secs))

    modelName = "classicSeqToSeqTSP"

    plot_train_and_validation_loss(epoch,
                                   losses_per_epoch,
                                   val_losses_per_epoch,
                                   experiment_details,
                                   num_nodes,
                                   modelName)

    plot_train_and_validation_reward(epoch,
                                     advantages_per_epoch,
                                     val_advantages_per_epoch,
                                     experiment_details,
                                     num_nodes,
                                     folder_name=modelName)

    f = open("training_metricsTSP.txt", "a")
    f.write(f"\n----------------- new experiment {get_filename_time()}\n")
    f.write(f"Experiment name: {experiment_details}\n")
    f.write(f"Training for {training_time_in_secs} seconds, minutes: {training_time_in_secs // 60}\n")
    f.write(f"Training for {epochs} and stopped at epoch {epoch}\n")

    return model, tours


if __name__ == '__main__':
    epochs =50
    num_nodes = 5
    train_size = 900000  # 000
    test_size = 10000
    validation_batch_size = 25
    batch_size = 256
    lr = 1e-4

    train_dataset = TSPDataset(train_size, num_nodes)
    test_dataset = TSPDataset(test_size, num_nodes)

    experiment_details = f'CorrectAdvantageTSPseq2seq_epochs{epochs}_train{train_size}_seqLen{num_nodes}_batch{batch_size}_lr{lr}'

    trainClassicSeq2SeqTSPWithReinforcementLearning(train_dataset,
                                                    test_dataset,
                                                    epochs,
                                                    experiment_details,
                                                    batch_size,
                                                    validation_batch_size,
                                                    num_nodes,
                                                    lr)
