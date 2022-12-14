import torch
import cv2 as cv
import numpy as np
from resnet import ResNet18
import os
import pickle as pkl


def predict_one(img_path, depth_path, d_layer, model_path, device):
    
    mean = [0.485, 0.456, 0.406, 0.4]
    std = [0.229, 0.224, 0.225, 0.2]
    mean = np.array(mean).reshape((4, 1, 1))
    std = np.array(std).reshape((4, 1, 1))

    img = cv.imread(img_path)  # 3, 224, 224
    img = np.transpose(img, [2, 0, 1])
    img = img / 255.
    d = np.load(depth_path, allow_pickle=True)[d_layer]  # 224 224
    d = d / np.max(d)
    d = np.expand_dims(d, 0)  # 1, 244, 244
    x = np.concatenate((img, d), 0)  # 4, 224, 224
    x = (x - mean) / std
    x = torch.unsqueeze(torch.tensor(x, dtype=torch.float64), 0).to(device)

    # model = torch.load(model_path)

    # This is a companion to save_state_dict(), which only saves the parameters, so the prediction has to be new model first
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path))

    output = model(x) # 1, 12

    return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './model/net_009.pth'

    img_path = './lazydata/test/X/0/rgb/0.png'
    depth_path = './lazydata/test/X/0/depth.npy'
    print(predict_one(img_path, depth_path, 0, model_path, device))


