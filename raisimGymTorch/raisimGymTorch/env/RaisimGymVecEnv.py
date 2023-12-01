# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os

import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def make_solo8(data, data_joint_pos, data_joint_vel):
    p_targets = np.zeros((len(data), 15))
    d_targets = np.zeros((len(data), 14))

    p_targets[:,:7] = data[:,:7]
    p_targets[:,7:] = data_joint_pos[:,:]
    d_targets[:,:6] = data[:,15:21]
    d_targets[:,6:] = data_joint_vel[:,:]

    return np.concatenate([p_targets, d_targets], axis=1)


def format_reference(data, chosen_feats = None):
    """
    data: reference motion traj
    """
    mask = [0,1,2,3,4,5,6,8,9,11,12,14,15,17,18,19,20,21,22,23,24,26,27,29,30,32,33,35,36]

    data = data[:,mask]
    data_joint_pos = data[:,7:15]
    data_joint_vel = data[:,21:]

    dataRR = data_joint_pos[:,0:2]
    dataRR_vel = data_joint_vel[:,0:2]
    dataRL = data_joint_pos[:,2:4]
    dataRL_vel = data_joint_vel[:,2:4]
    dataFR = data_joint_pos[:,4:6]
    dataFR_vel = data_joint_vel[:,4:6]
    dataFL = data_joint_pos[:,6:]
    dataFL_vel = data_joint_vel[:,6:]

    data_joint_pos = np.concatenate((dataFL,dataFR,dataRL,dataRR),axis=-1)
    data_joint_vel = np.concatenate((dataFL_vel,dataFR_vel,dataRL_vel,dataRR_vel),axis=-1)

    solo8_data = make_solo8(data, data_joint_pos, data_joint_vel)

    if chosen_feats is not None:
        solo8_data = solo8_data[:, chosen_feats]
    
    return solo8_data


class RaisimGymVecEnv:

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._ref = np.zeros((self.num_envs, 37))  #NOTE: dim = (n_envs, n_columns) of the reference csv file 
        self.actions = np.zeros([self.num_envs, self.num_acts], dtype=np.float32)
        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.observation_space = np.zeros(self.num_obs)
        self.action_space = np.zeros(self.num_acts)
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.num_obs, dtype=np.float32)
        self.var = np.zeros(self.num_obs, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observation, update_statistics)
        return self._observation
    
    def getref(self):
        self.wrapper.getref(self._ref)
        ref = format_reference(self._ref)
        return ref
        

    def reset(self):
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class RaisimGymVecTorchEnv:

    def __init__(self, impl, cfg, normalize_ob=False, seed=0, normalize_rew=False, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.observation_space = np.zeros(self.num_obs)
        self.action_space = np.zeros(self.num_acts)

        self._observation_torch = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float32, device=device)
        self._reward_torch = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self._done_torch = torch.zeros(self.num_envs, dtype=torch.int32, device=device)

        self.total_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=device)

        # time limit
        self.t = torch.zeros(self.num_envs, dtype=torch.int32)
        self.time_limit = 150

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action.to("cpu").numpy(), self._reward, self._done)
        self.wrapper.observe(self._observation, False)
        self._reward_torch[:] = torch.from_numpy(self._reward[:]).to(device)
        self._done_torch[:] = torch.from_numpy(self._done[:]).to(device)
        self.get_total_reward()
        return self.observe(), self._reward_torch[:], self._done_torch[:], {}

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.obs_rms.count = count
        self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self):
        self.wrapper.observe(self._observation, False)

        self._observation_torch[:, :] = torch.from_numpy(self._observation[: ,:]).to(device)
        return self._observation_torch
    



    def reset(self):
        self._reward[:] = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def _normalize_observation(self, obs):
        if self.normalize_ob:

            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        else:
            return obs

    def reset_time_limit(self):
        self.wrapper.timeLimitReset()

    def get_total_reward(self):
        total_rewards = self.wrapper.getTotalRewards();
        self.total_rewards[:] = torch.from_numpy(total_rewards).to(device)

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

