import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np


print(suite.ALL_GRIPPERS)

robot = 'UR5e'
gripper = 'Robotiq85Gripper'
options = dict()
options['env_name'] = 'Lift'
options['robots'] = robot
options['controller_configs'] = [load_controller_config(default_controller='OSC_POSE')]
options['gripper_types'] = gripper
env = suite.make(**options, has_renderer=True, ignore_done=True, use_camera_obs=True, camera_names='frontview')

env.reset()

for i in range(100):
    action = np.random.randn(env.action_dim) # sample random action
    print(action)
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

env.close()