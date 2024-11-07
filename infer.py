import argparse
import os
from PIL import Image
import cv2
import torch
import numpy as np
import imageio
from model import get_model, get_data, get_web_img

def get_jave_7action(key):
    if key == "r":
        action = 1
    elif key == "rj":
        action = 2
    elif key == "l":
        action = 3
    elif key == "lj":
        action = 4
    elif key == "j":
        action = 5
    elif key == "f":
        action = 6
    else:
        action = 0
    return [action]

def get_action_sequence(actions):
    ret = []
    for a in actions:
        assert a in ['l', 'r', 'f', 'j', 'lj', 'rj', 'n'], f"action {a} should be in [l->left r->right j->jump f->fire lj->left jump rj->right jump n->null]"
        ret.extend(get_jave_7action(a))
    return ret


def preprocess_img(img):
    target_size = (128, 128)
    cropped_image = cv2.resize(img[:, :, :3], target_size) 
    final_image = np.expand_dims(np.expand_dims(cropped_image.transpose(2, 0, 1), axis=0), axis=0)
    final_image = torch.tensor(final_image, dtype=torch.float32, device=model.device)
    batch_data = {'observations': final_image}
    return batch_data

def image_to_numpy_array(filepath):
    img = Image.open(filepath)
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return img_array

obs_shape = [3,128,128]

def get_img_data(img_path, device):
    img = Image.open(img_path)
    img = np.array(img)
    img_data = img[np.newaxis,...] # (1, 128,128,3)
    
    cur_img = torch.tensor(img_data, dtype=torch.float32, device=device)  
    
    cur_img = cur_img.permute(0, 3, 1, 2)
    cur_img = cur_img.reshape(1, obs_shape[0], obs_shape[1], obs_shape[2])
    cur_img = cur_img / 255.0
    return cur_img

def init_simulator(model, batch):
    obs = batch["observations"]
    with torch.no_grad():
        wm_env = model.init_wm(obs)
        return obs, wm_env


def parse_comma_separated_list(value):
    return value.split(',')

def arg():
    parser = argparse.ArgumentParser(description="Direct inference of Playable Game Generation")

    parser.add_argument('-i', "--img", type=str, required=True, help="The initial screen of the game")
    parser.add_argument('-a', "--actions", required=True, type=parse_comma_separated_list, help="action sequences\n l->left r->right j->jump f->fire lj->left jump rj->right jump n->null")
    parser.add_argument('-s', "--sample_step", default=4, type=int, help="diffusion sample step")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg()
    actions = get_action_sequence(args.actions)
    sample_step = args.sample_step

    img_list = []
    model = get_model()


    batch_data = {}
    batch_data["observations"] = get_img_data(args.img, model.device)
    with torch.no_grad():
        obs, zeta = init_simulator(model, batch_data)
    img_list.append(get_web_img(obs[0].cpu().numpy()))

    for a in actions:
        with torch.no_grad():
            obs, zeta = model.real_time_infer(zeta, torch.tensor([a]).long(), sample_step)
        img_list.append(get_web_img(obs[0].cpu().numpy()))


    if not os.path.isdir('./output/'):
        os.makedirs('./output/')
    imageio.mimsave('./output/output.gif', img_list, duration=0.1)
