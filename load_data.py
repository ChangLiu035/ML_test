import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2 as cv
import os
import random


def get_x_path(path):
    l = os.listdir(path + '/X')
    sample_num = len(l)
    start = l[0]
    img_path = []
    d_path = []
    id_path = []
    for i in range(int(start), int(start)+sample_num):
        for j in range(3):
            img_path.append(path + '/X/' + str(i) + '/rgb/' + str(j) + '.png')
            d_path.append((path + '/X/' + str(i) + '/depth.npy', j))
            id_path.append(path + '/X/' + str(i) + '/field_id.pkl')
    return img_path, d_path, id_path


def get_y_path(path):
    l = os.listdir(path + '/Y')
    sample_num = len(l)
    start = l[0].split('.')[0]
    y_path = []
    for i in range(int(start), int(start)+sample_num):
        for j in range(3):
            y_path.append(path + '/Y/' + str(i) + '.npy')
    return y_path


class FingerTip(Dataset):
    def __init__(self, path, mean, std, train=True, flip=True):
        self.img_path, self.d_path, self.id_path = get_x_path(path)

        self.mean = np.array(mean).reshape((4, 1, 1))
        self.std = np.array(std).reshape((4, 1, 1))

        self.train = train

        self.flip = flip

        if train:
            self.y_path = get_y_path(path)

    def __getitem__(self, index):
        img = cv.imread(self.img_path[index])  # 3, 224, 224
        img = np.transpose(img, [2, 0, 1])
        img = img / 255.
        d = np.load(self.d_path[index][0], allow_pickle=True)[self.d_path[index][1]] # 224 224
        d = d/np.max(d)
        d = np.expand_dims(d, 0) # 1, 244, 244
        x = np.concatenate((img, d), 0)  # 4, 224, 224
        x = (x - self.mean) / self.std
        if self.flip:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    np.flip(x, 1)
                else:
                    np.flip(x, 2)
        x = torch.tensor(x, dtype=torch.float64)
        if self.train:
            y = np.load(self.y_path[index], allow_pickle=True)  # 12,
            y = torch.tensor(y, dtype=torch.float64)
            return x, y
        return x

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dir = "./lazydata/vali"
    test_dir = "./lazydata/train"

    mean = [0.485, 0.456, 0.406, 0.4]
    std = [0.229, 0.224, 0.225, 0.2]

    train_set = FingerTip(train_dir, mean, std)
    test_set = FingerTip(test_dir, mean, std, False)

    batch_size = 16
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=batch_size)

    # print(len(train_iter))
    for xx, yy in train_iter:
        xx, yy = xx.to(device), yy.to(device)
        print(xx.shape, yy.shape)
