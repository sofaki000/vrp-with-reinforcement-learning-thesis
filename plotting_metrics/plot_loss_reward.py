import os

import torch
import math
import numpy as np
import datetime

from IPython.core.display_functions import clear_output
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix


def create_distance_matrix(points):
    return distance_matrix(points, points)

def create_distance_matrix_for_batch_elements(static):
    '''
    static: [batch_size, static_features=2, number_of_nodes]
    '''
    distances_per_batch = []
    for batch in static:
        ds_matrix = create_distance_matrix(batch.transpose(1,0))
        distances_per_batch.append(ds_matrix)

    return distances_per_batch



def get_filename_time():
    now = datetime.datetime.now()
    return f'm={now.month}_d={now.day}_h={now.hour}_m={now.minute}'

def show_tour_for_model(ax, distance_matrix, nodes, tour):
    N = nodes.size(0)
    distance =0.
    start_node = 0

    for i in range(N):
        start_pos = nodes[start_node]
        if i != N-1:
            next_node = tour[i + 1]
        else:
            next_node = tour[0]

        end_pos = nodes[next_node]

        ax.annotate(text="",xy=start_pos, xycoords='data',
                           xytext=end_pos, textcoords='data',
                           arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))

        #print(f'{start_node}->{next_node}, dist:{distance_matrix[start_node][next_node]}')
        distance += distance_matrix[start_node][next_node] #math.dist(end_pos, start_pos)
        start_node = next_node
        if torch.is_tensor(end_pos[0]):
            if torch.is_tensor(next_node):
                ax.text(end_pos[0].item(), end_pos[1].item(), next_node.item(), size=10, color='r')
            else:
                ax.text(end_pos[0].item(), end_pos[1].item(), next_node, size=10, color='r')
        else:
            ax.text(end_pos[0], end_pos[1], next_node, size=10, color='b')

    textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
               fontsize=10,  # Textbox
               verticalalignment='top', bbox=props)

def get_2city_distance(n1, n2):
	x1,y1,x2,y2 = n1[0],n1[1],n2[0],n2[1]
	if isinstance(n1, torch.Tensor):
		return torch.sqrt((x2-x1).pow(2)+(y2-y1).pow(2))
	elif isinstance(n1, (list, np.ndarray)):
		return math.sqrt(pow(x2-x1,2)+pow(y2-y1,2))
	else:
		raise TypeError

def show_tour(nodes, distance_matrix, model_tour, or_tour, filename, model_tour_title):
    '''
    nodes: tensor [num_nodes, 2]
    model_tour, or_tour: [num_nodes+1]
    '''
    assert nodes.size(1) == 2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, sharex=True, sharey=True)  # Prepare 2 plots
    ax[0].set_title('OR tour')
    ax[0].scatter(nodes[:, 0], nodes[:, 1])  # plot A
    show_tour_for_model(ax[0],distance_matrix, nodes, or_tour)


    ax[1].set_title(model_tour_title)
    ax[1].scatter(nodes[:, 0], nodes[:, 1])  # plot B
    show_tour_for_model(ax[1],distance_matrix, nodes, model_tour)

    plt.tight_layout()

    plt.savefig(f"tours\\tour_{filename}_{get_filename_time()}.png")

def show_tour_for_one_solution(nodes, distance_matrix,   or_tour, filename ,title):
    '''
    nodes: tensor [num_nodes, 2]
    model_tour, or_tour: [num_nodes+1]
    '''
    assert nodes.size(1) == 2
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, sharex=True, sharey=True)  # Prepare 2 plots
    ax.set_title(title)
    ax.scatter(nodes[:, 0], nodes[:, 1])  # plot A
    show_tour_for_model(ax,distance_matrix, nodes, or_tour)

    plt.tight_layout()

    plt.savefig(f"{filename}.png")

def plot_train_loss_and_train_reward(epoch, train_loss, train_reward,experiment_details="", folder_name=None):
    title1 = f"Train loss epoch {epoch}, {train_loss[-1]:.2f}"
    title2 = f"Train reward epoch {epoch},{train_reward[-1]:.2f}"
    filename = f"train_metrics_{experiment_details}_date{get_filename_time()}.png"

    os.makedirs(folder_name, exist_ok=True)
    #plot_train_and_validation_metrics("Loss", train_loss, train_reward, title1,title2, filename,folder_name)
    clear_output(True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # # Add x and y labels to all subplots
    # for ax in axes.flat:
    #     ax.set(xlabel='Epochs', ylabel=f'{metric_name}')

    axes[0].plot(train_loss)
    axes[0].set_title(title1)
    axes[0].set(xlabel='Epochs', ylabel="Loss")
    axes[0].grid()
    axes[1].grid()
    axes[1].plot(train_reward)
    axes[1].set(xlabel='Epochs', ylabel="Reward")
    axes[1].set_title(title2)

    # directory =f'metrics\\{epoch}'
    now = datetime.datetime.now()
    directory = f'metrics\\day_{now.day}\\hour_{now.hour}'

    if folder_name is not None:
        directory = f'metrics\\day_{now.day}\\hour_{now.hour}\\{folder_name}'

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\{filename}')

    plt.clf()

def plot_train_and_validation_loss(epoch, train_loss, val_loss,experiment_details="", num_nodes=None,folder_name=None):
    title1 = f"Train loss epoch {epoch}, Nodes:{num_nodes}"
    title2 = f"Val loss epoch {epoch}"
    filename = f"losses_{experiment_details}_date{get_filename_time()}.png"
    plot_train_and_validation_metrics("Loss", train_loss, val_loss, title1,title2, filename,folder_name)


def plot_train_and_validation_distance(epoch, train_reward, val_reward,experiment_details="", num_nodes=None,folder_name=None):
    title1 = f"Distances at training. Stopped at epoch {epoch}"
    title2 = f"Distances at validation. Stopped at epoch {epoch}"
    filename = f"rewards_{experiment_details}_date{get_filename_time()}.png"
    os.makedirs(folder_name, exist_ok=True)
    plot_train_and_validation_metrics("Distance", train_reward, val_reward, title1,title2, filename,folder_name)


def plot_train_and_validation_reward(epoch, train_reward, val_reward,experiment_details="", num_nodes=None,folder_name=None):
    title1 = f"Train reward epoch {epoch}, Nodes:{num_nodes}"
    title2 = f"Val reward epoch {epoch}"
    filename = f"rewards_{experiment_details}_date{get_filename_time()}.png"
    os.makedirs(folder_name, exist_ok=True)
    plot_train_and_validation_metrics("Reward", train_reward, val_reward, title1,title2, filename,folder_name)


def plot_reward(rewards, title, filename,folder_name=None):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    now = datetime.datetime.now()
    directory = f'metrics\\day_{now.day}'

    if folder_name is not None:
        directory = f'metrics\\day_{now.day}\\{folder_name}'

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\{filename}.png')
    plt.clf()



def plot_train_and_validation_metrics(metric_name,train_metric, val_metric,
                                      title1,title2,
                                      filename,
                                      folder_name=None):
    clear_output(True)
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
    directory = f'metrics\\day_{now.day}\\hour_{now.hour}\\min_{now.minute}'

    if folder_name is not None:
        directory = f'metrics\\day_{now.day}\\hour_{now.hour}\\min_{now.minute}\\{folder_name}'

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\{filename}')

    plt.clf()

def plot_train_tour_length(seq_len, epoch,train_tour,val_tour,train_size,batch_size, epochs ):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)


    plt.title('train tour length: epoch %s reward %s' % (epoch, train_tour[-1] if len(train_tour) else 'collecting'))
    plt.plot(train_tour)
    plt.grid()
    plt.ylabel("Tour length")
    plt.subplot(132)
    plt.title(
        'val tour length: epoch %s reward %s' % (epoch, val_tour[-1] if len(val_tour) else 'collecting'))
    plt.plot(val_tour)
    plt.grid()

    experiment_details = f'epochs{epochs}_seqLen{seq_len}_train{train_size}_batch{batch_size}'
    now = datetime.datetime.now()
    directory = f'metrics\\day_{now.day}\\hour_{now.hour}'

    os.makedirs(directory, exist_ok=True)
    plt.savefig(f'{directory}\\results{experiment_details}_{get_filename_time()}.png')


