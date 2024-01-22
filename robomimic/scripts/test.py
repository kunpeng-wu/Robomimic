# if you want to use robosuite in PycharmProjects/Robosuite, you should add the python path into .py through sys.path.append
import sys
sys.path.append("/home/kavin/Documents/PycharmProjects/Robosuite")

import robosuite

print(robosuite.__logo__)

import collections
import time
import datetime
import h5py
import numpy as np

from threading import Thread, Lock

def func(list_of_dict):
    dic = collections.OrderedDict()
    for i in range(len(list_of_dict)):
        for k in list_of_dict[i]:
            if k not in dic:
                dic[k] = []
            dic[k].append(list_of_dict[i][k])
    return dic


obs_list = []
for i in range(2):
    obs = {"a": i * 100, "b": i + 200}
    obs_list.append(obs)
print(obs_list)

obs_dict = func(obs_list)
print(obs_dict)

t_now = time.time()
time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')
print(time_str)

obs_modality_mapping = {}
obs_keys = ['robot0_eef_qpos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object', 'object']
obs_modality_mapping['low_dim'] = obs_keys

OBS_KEYS_TO_MODALITIES = {}
for obs_key in obs_keys:
    if obs_key not in OBS_KEYS_TO_MODALITIES:
        OBS_KEYS_TO_MODALITIES[obs_key] = 'low_dim'

print(OBS_KEYS_TO_MODALITIES)

print(obs_modality_mapping)
OBS_MODALITIES_TO_KEYS = { obs_modality : list(set(obs_modality_mapping[obs_modality])) for obs_modality in obs_modality_mapping }
print(OBS_MODALITIES_TO_KEYS)

f = h5py.File("/home/kavin/Documents/PycharmProjects/Robomimic/datasets/lift/humanoid/low_dim.hdf5", "r")
filter_key = "valid"
demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
f.close()
print(demo_keys)

step_log_dict = {}
step_log_dict['l1_loss'] = [1, 2, 3, 4, 5]
# step_log_all = {}
# for k, v in step_log_dict.items():
#     step_log_all[k] = float(np.mean(v))
step_log_all = dict((k, np.mean(v))for k, v in step_log_dict.items())
print(step_log_all)

rollout_info_1 = {"Return": 12, "Horizon": 400, "Success_Rate": 1}
rollout_info_2 = {"Return": 24, "Horizon": 300, "Success_Rate": 1}
rollout_logs = [rollout_info_1, rollout_info_2]
rollout_logs = dict((k, [rollout_logs[i][k] for i in range(len(rollout_logs))]) for k in rollout_logs[0])
print(rollout_logs)

demos = ['demo_5', 'demo_4', 'demo_1']
idx = np.argsort([int(elem[5:]) for elem in demos])     # [2 1 0]
demos = [demos[i] for i in idx]
# print(demos)    # ['demo_1', 'demo_4', 'demo_5']

import os
path = os.path.split(robosuite.__file__)[0]
path_split = path.split("/")
old_path = "robosuite/models/assets/robots/humanoid/meshes/left_1.STL"
old_path_split = old_path.split("/")
ind = max(loc for loc, val in enumerate(old_path_split) if val == "robosuite")  # last occurrence index
new_path_split = path_split + old_path_split[ind + 1:]
print(new_path_split)
new_path = "/".join(new_path_split)
print(new_path)

# TODO: hdf5 file
dataset = "/home/kavin/Documents/PycharmProjects/Robomimic/datasets/lift/test/demo.hdf5"
f = h5py.File(dataset, "r")
print(list(f.keys()))   # ['data', 'mask']
print(list(f["data"].keys()))   # ['demo_1']
print(list(f["mask"].keys()))   # ['train']
print(f["data"].attrs["env"])
print(f["data"].attrs["env_info"])
# print(f["data"].attrs["env_args"])
print(f["data"].attrs["repository_version"])
print(f["data"].attrs["total"])

# for ep in f["data"]:
#     actions = f["data/{}/actions".format(ep)][()]
#     states = f["data/{}/states".format(ep)][()]
#     print(actions.shape, states.shape)
    # print(f["data/{}".format(ep)].attrs["num_samples"])
    # print(f["data/{}".format(ep)].attrs["model_file"])
states = f["data/demo_1/states"][()]
init_state = dict(state=states[0])
print(init_state)
# k = "mask/train"
# f[k] = np.array(['demo_1'], dtype='S')
f.close()