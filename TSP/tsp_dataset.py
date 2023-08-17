from scipy.spatial import distance_matrix
import numpy as np
import itertools
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


distance_matrix_table = []


def create_distance_matrix(points):
    return distance_matrix(points, points)



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
        sample = {'Points':tensor }
        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
            points_list.append(np.random.random((self.seq_len, 2)))#np.random.randint(30,size=(self.seq_len, 2))) # np.random.random((self.seq_len, 2))

        return {'Points_List':points_list }

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec