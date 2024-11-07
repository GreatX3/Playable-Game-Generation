import importlib
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd
import random
from einops import rearrange
from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageDraw, ImageFont
import pickle
from scipy.ndimage import zoom
import imageio
from network.df.config.Config import Config

def read_model(model_name, ckpt_name, action_space, device='cpu'):
    if "iris" in model_name:
        model = importlib.import_module(
            "network." + "iris" + ".algorithm").Algorithm(model_name, action_space)
    elif "df" in model_name:
        model = importlib.import_module(
            "network." + "df" + ".algorithm").Algorithm(model_name, device)
    else:
        raise NotImplementedError
    state_dict = torch.load(
        osp.join("ckpt", ckpt_name, "model.pth"),
        map_location=torch.device(device)
    )
    model.load_state_dict(state_dict['network_state_dict'],strict=False)
    model.eval().to(device)
    return model

def read_file(data_file, data_type="java"):
    print("read_file", data_type)
    data_with_nan = np.genfromtxt(data_file, delimiter=',')
    np_data = data_with_nan[:, ~np.isnan(data_with_nan).any(axis=0)]
    data = np_data[:, Config.data_start_end[0]:Config.data_start_end[1]]
    return data

def split_data_mario(epi_data, horizon_len, device):
    episode_len, _ = epi_data.shape
    assert episode_len % horizon_len == 0
    data_tensor = torch.tensor(epi_data, dtype=torch.float32, device=device)
    cur_img = data_tensor[:, :-1].reshape(episode_len, Config.resolution, Config.resolution, 3)
    cur_img = cur_img.permute(0, 3, 1, 2)
    cur_img = cur_img.reshape(episode_len, 3, Config.resolution, Config.resolution)
    cur_img = cur_img / 255.0
    cur_action = data_tensor[:, -1].reshape(episode_len, 1).long()
    dict_data = {
        "epi_len": episode_len,
        "cur_img": cur_img,
        "cur_act_int": cur_action,
    }
    return dict_data
