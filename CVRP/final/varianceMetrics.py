import torch
from torch.utils.data import DataLoader

from CVRP.final.official_cvrp_notebook import Seq2SeqModelCVRPWithAttention, VehicleRoutingDataset, train, reward, \
    render,  validate

STATIC_SIZE = 2  # (x, y)
DYNAMIC_SIZE = 2
hidden_size = 128
actor_lr = 5e-3
critic_lr = 5e-4  # 0.0005
max_grad_norm = 2.
hidden = hidden_size
step = 5
dropout = 0.1
layers = 1
gamma = 0.1
LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
MAX_DEMAND = 9
train_size = 1_000  # 0  # 000  # 0#000  # 1000000
bs = 5  # 256
validation_bs = 5
validation_size = 20  # 0  # 0#0
hidden_size = 128
epochs = 20  # 0
seq_len = 10

max_load = LOAD_DICT[seq_len]

valid_data = VehicleRoutingDataset(validation_size,
                                   seq_len,
                                   max_load,
                                   MAX_DEMAND)
max_load = LOAD_DICT[10]
train_data = VehicleRoutingDataset(12,
                                   20,
                                   max_load,
                                   MAX_DEMAND,
                                   12345)


def variance_of_weights(model):
    variances = {}

    for name, param in model.named_parameters():
        if 'weight' in name:
            variances[name] = torch.var(param.data)

    # Formatting the output
    formatted_output = "Model Weights Variance:\n"
    for key, value in variances.items():
        formatted_output += f"- {key}: {value.item():.4f}\n"

    return formatted_output


def variance_of_predictions(model, dataloader):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            static, dynamic, x0 = batch
            # Full forward pass through the dataset
            tour_indices, tour_logp = model(static, dynamic, x0)
            for tour_log in tour_logp:
                all_preds.extend(tour_log.tolist())

    mean_preds = sum(all_preds) / len(all_preds)
    variance = sum([(p - mean_preds) ** 2 for p in all_preds]) / len(all_preds)
    return variance

train_size = 1_000  # 0  # 000  # 0#000  # 1000000
bs = 5  # 256
validation_bs = 5
validation_size = 20  # 0  # 0#0
hidden_size = 128
epochs = 20  # 0
seq_len = 10

def performance_variance( valid_loader, num_runs=4):
    performances = []

    for i in range(num_runs):

        model = Seq2SeqModelCVRPWithAttention(STATIC_SIZE,
                                              DYNAMIC_SIZE,
                                              hidden_size,
                                              train_data.update_dynamic,
                                              train_data.update_mask,
                                              num_layers=1,
                                              dropout=0.1)

        experiment_details = f'VARIANCE_TEST_epoch{epochs}_trainSize{train_size}_bs{bs}_valSize{validation_size}_hidden{hidden_size}_actlr{actor_lr}'

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
              max_grad_norm,
              modelName="VARIANCE",
              task="cvrp_variance",
              verbose=False)

        mean_valid_reward, mean_valid_loss, mean_validation_distance = validate(valid_loader, model, reward)

        performances.append(mean_valid_reward)

    mean_performance = sum(performances) / len(performances)
    variance_performance = sum([(p - mean_performance) ** 2 for p in performances]) / len(performances)
    print(variance_performance)

    return variance_performance


def loadModel():
    model = Seq2SeqModelCVRPWithAttention(STATIC_SIZE,
                                          DYNAMIC_SIZE,
                                          hidden_size,
                                          train_data.update_dynamic,
                                          train_data.update_mask,
                                          num_layers=1,
                                          dropout=0.1)
    PATH = "../cvrp_modelAttentionCVRPNormalizingR_epoch30_trainSize10000_bs8_valSize300_hidden128_lr0.0005.pt"
    model.load_state_dict(torch.load(PATH), strict=False)

    return model


if __name__ == '__main__':
    model = loadModel()
    batch_size = 2


    dataloader = DataLoader(valid_data, batch_size, False, num_workers=0)

    variance = variance_of_predictions(model, dataloader)
    print(f"Variance of predictions: {variance}")

    variances = variance_of_weights(model)
    print(variances)

    variances = performance_variance(dataloader, num_runs=4)

    print(f'Performance variance: {variances}')
