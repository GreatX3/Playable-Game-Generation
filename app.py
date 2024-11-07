from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import queue
from threading import Thread, Event
import torch
import time

from model import get_model, get_data, init_simulator, get_web_img

"""Model info"""
frame_rate = 1/20

model = get_model()
init_data = get_data()
# with torch.no_grad():
#     init_obs, init_zeta = init_simulator(model, init_data)


app = Flask(__name__)
socketio = SocketIO(app)

"""User Info"""
user_zeta = {}
user_numpy_data = {}
user_queues = {}
user_cmd = {}
user_config = {}
user_threads = {}
online_player = {}

default_cmd_str = "None"

def numpy2imgstr(image):
    image = get_web_img(image)
    img = Image.fromarray(image)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

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
    elif key == "s":
        action = 6
    else:
        action = 0
    return [action]

def get_user_config(data):
    ret = {
        'random_init': False,
        'block': False,
        'denosing_step': 4,
    }
    random_init = data['random_init']
    block = data['block']
    denosing_step = data['denosing_step']
    if random_init == 'true':
        ret['random_init'] = True
    if block == 'true':
        ret['block'] = True
    if denosing_step == '8':
        ret['denosing_step'] = 8
    return ret

@socketio.on('key_press')
def handle_key_press(data):
    user_id = request.sid
    key = data['key']
    user_cmd[user_id] = key

@socketio.on('key_release')
def handle_key_release(data):
    user_id = request.sid
    key = data['key']
    if user_cmd[user_id] == 'lj':
        if key == 'l':
            user_cmd[user_id] = 'j'
        elif key == 'j':
            user_cmd[user_id] = 'l'
    elif user_cmd[user_id] == 'rj':
        if key == 'r':
            user_cmd[user_id] = 'j'
        elif key == 'j':
            user_cmd[user_id] = 'r'
    else:
        user_cmd[user_id] = default_cmd_str

    


@socketio.on('start_game')
def button_clicked(data):
    user_id = request.sid
    player_config = get_user_config(data)
    user_config[user_id] = player_config
    if player_config['random_init']:
        random_data = get_data(if_random=True)
        user_numpy_data[user_id] = random_data
    else:
        user_numpy_data[user_id] = init_data

    online_player[user_id] = user_id
    print(f"UserId:{user_id} start game with config: {player_config}")
    socketio.emit('update_person', {'num': len(online_player)})

def model_inference(user_id, stop_event):
    while not stop_event.is_set():
        if user_id in user_numpy_data.keys():
            start_time = time.time()
            with torch.no_grad():
                obs, zeta = init_simulator(model, user_numpy_data[user_id])
            user_zeta[user_id] = zeta
            end_time = time.time()
            obs = obs[0].cpu().numpy()
            duration = f"{end_time - start_time:.2f} second"
            rest_time = frame_rate - (end_time - start_time)
            if rest_time > 0:
                time.sleep(rest_time)

            user_queues[user_id].put((obs, "None", duration))
            user_numpy_data.pop(user_id, None)
        elif user_id in user_cmd.keys() and user_id in user_zeta.keys():
            start_time = time.time()
            key = user_cmd[user_id]
            block = user_config[user_id]['block']
            sampling_timesteps = user_config[user_id]['denosing_step']
            if key == default_cmd_str and block:
                continue

            action = get_jave_7action(key)
            action = torch.Tensor(action).long()
            zeta = user_zeta[user_id]
            with torch.no_grad():
                obs, zeta = model.real_time_infer(zeta, action, sampling_timesteps)
            user_zeta[user_id] = zeta
            end_time = time.time()
            obs = obs[0].cpu().numpy()
            duration = f"{end_time - start_time:.2f} second"
            rest_time = frame_rate - (end_time - start_time)
            if rest_time > 0:
                time.sleep(rest_time)

            user_queues[user_id].put((obs, key, duration))
        else:
            time.sleep(0.01)


def send_results(user_id, stop_event):

    while not stop_event.is_set():
        if not user_queues[user_id].empty():
            obs, cmd_str_, dur = user_queues[user_id].get()
            socketio.emit('update_frame', {
                'image': numpy2imgstr(obs),
            }, room=user_id)
            socketio.emit('update_cmd', {'cmd': cmd_str_, 'dur': dur}, room=user_id)
        else:
            time.sleep(0.01)

@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    print(f"connect userId added: {user_id}")
    user_cmd[user_id] = default_cmd_str
    user_queues[user_id] = queue.Queue()
    join_room(user_id)
    
    stop_event = Event()
    inference_thread = Thread(target=model_inference, args=(user_id, stop_event))
    inference_thread.daemon = True
    inference_thread.start()
    
    result_thread = Thread(target=send_results, args=(user_id, stop_event))
    result_thread.daemon = True
    result_thread.start()

    user_threads[user_id] = (inference_thread, result_thread, stop_event)
    socketio.emit('update_person', {'num': len(online_player)})

@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    leave_room(user_id)
    user_cmd.pop(user_id, None)
    user_queues.pop(user_id, None)
    user_config.pop(user_id, None)
    online_player.pop(user_id, None)
    if user_id in user_threads:
        inference_thread, result_thread, stop_event = user_threads.pop(user_id)
        stop_event.set()
        inference_thread.join()
        result_thread.join()
    print(f"disconnect userId added: {user_id}")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8080)
