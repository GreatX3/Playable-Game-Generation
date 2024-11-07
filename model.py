import utils
import random
import torch
from einops import rearrange
import time
import numpy as np
import cv2
from config import *

def process_npdata(np_data, horizon_len, start_idx, device):
    file_data_len = np_data.shape[0]
    end_idx = start_idx + horizon_len
    if end_idx > file_data_len:
        end_idx = file_data_len
        start_idx = end_idx - horizon_len
    epi_data = np_data[start_idx:end_idx]
    dict_data = utils.split_data_mario(epi_data, horizon_len, device)
    batch = {}
    for k, v in dict_data.items():
        if torch.is_tensor(v):
            batch[k] = v
    batch_data = data_formater(batch, horizon_len)
    batch_data["start_idx"] = start_idx
    return batch_data

def data_formater(data_dict, epi_len):
    batch = {}
    ori_obs = data_dict["cur_img"]
    batch["observations"] = rearrange(ori_obs, '(b t) c h w -> b t c h w', t=epi_len)
    batch["cur_actions"] = rearrange(data_dict["cur_act_int"], '(b t) e -> b t e', t=epi_len, e=1)
    return batch

def init_simulator(model, batch):
    obs = batch["observations"]
    obs = rearrange(obs, 'b t c h w -> (b t) c h w')
    with torch.no_grad():
        init_zeta = model.init_wm(obs[0:1])
        return obs, init_zeta

def get_web_img(img):
    # img.shape = [c, h, w]
    img_3ch = np.transpose(img, (1,2,0)) # [h, w, c]
    img_3ch = np.clip(img_3ch, 0, 1)
    img_3ch = cv2.resize(img_3ch, (300, 300), interpolation=cv2.INTER_LINEAR)
    # img_3ch = cv2.resize(img_3ch, (25, 25), interpolation=cv2.INTER_LINEAR)
    img_3ch = (img_3ch*255.0).astype(np.uint8)
    return img_3ch

def get_model():
    model = utils.read_model(model_name, model_path, action_space, device)
    return model

np_data = utils.read_file(file_path, data_type)

def get_data(if_random=False):
    start_time = time.time()
    start_idx = 0
    if if_random:
        start_idx = random.randint(0, np_data.shape[0])
    batch_data = process_npdata(np_data, 1, start_idx, device)
    return batch_data

if __name__ == "__main__":
    model = utils.read_model(model_name, model_path, action_space, device)

    np_data = utils.read_file(file_path, data_type)

    start_time = time.time()
    start_idx = random.randint(0, np_data.shape[0])
    start_idx = 0
    batch_data = process_npdata(np_data, 1, start_idx, device)
    end_time = time.time()
    print(f"process_npdata cost time: {end_time - start_time:.4f} s")
    start_time = time.time()
    data = batch_data["observations"]

    obs, wm = init_simulator(model, batch_data)
    img = get_web_img(obs[0].cpu().numpy())
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./eval_data/init.jpg', img)
    # print(img.shape)

