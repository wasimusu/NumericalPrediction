from torch.utils.data import DataLoader, Dataset


class RegressionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, id):
        sample = self.inputs[id], self.labels[id]
        return sample


if __name__ == '__main__':
    pass
