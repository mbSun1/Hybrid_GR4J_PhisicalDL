import numpy as np
import pandas as pd
from torch.utils.data import dataset, dataloader


class get_loader(object):
    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame, sequence_length: int,
                 batch_size: int, window_step: int = 365, spin_up_dayl: int = 365):
        self.sequence_length = sequence_length

        # get datasets for training, and test sets
        self.train_set, self.test_set = self._generate_sequences(train_set, test_set,window_step, spin_up_dayl)

        # get dataloaders for training, and test sets
        self.train_loader = dataloader.DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = dataloader.DataLoader(dataset=self.test_set, batch_size=100, shuffle=False)

    def _generate_sequences(self, train_set, test_set, window_step=365, spin_up_dayl=365):
        """
        train_set, valid_set, test_set: pandas.dataframe of training, validation, and test sets
        window_step: int, the skip length for the moving window
        warm_up_dayl: int, data length for warming up model to get stable initial states
        """
        # calculate sequence number based on moving window step and sequence length
        wrap_number_train = (train_set.shape[0] - self.sequence_length) // window_step + 1

        # split data into sequences for training set
        train = np.empty(shape=(wrap_number_train, self.sequence_length, train_set.values.shape[1]))
        for i in range(wrap_number_train):
            train[i, :, :] = train_set.values[i * window_step:(self.sequence_length + i * window_step), :]

        # add a warm-up period for test sets
        test = np.expand_dims(pd.concat([train_set[-spin_up_dayl:], test_set]).values, axis=0)
        train, test = my_dataset(train),  my_dataset(test)
        return train,  test


class my_dataset(dataset.Dataset):
    def __init__(self, data):
        self.x = data[:, :, :-1]
        self.y = data[:, :, -1:]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

