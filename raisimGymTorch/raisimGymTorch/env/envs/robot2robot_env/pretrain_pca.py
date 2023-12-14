import numpy as np
import os.path as osp
import glob
from sklearn.decomposition import PCA
import joblib
from ruamel.yaml import YAML


TASK_PATH = osp.dirname(osp.realpath(__file__))
HOME_PATH = TASK_PATH + "/../../../../.." + "/rsc"
DATA_DIR = "traj/solo8/"


def make_solo8(data, data_joint_pos, data_joint_vel):
    p_targets = np.zeros((len(data), 15))
    d_targets = np.zeros((len(data), 14))

    p_targets[:,:7] = data[:,:7]
    p_targets[:,7:] = data_joint_pos[:,:]
    d_targets[:,:6] = data[:,15:21]
    d_targets[:,6:] = data_joint_vel[:,:]

    return np.concatenate([p_targets, d_targets], axis=1)


def load_traj(path, chosen_feats = None):
    data = np.loadtxt(path,delimiter=",")
    # mask = [0,1,2,3,4,5,6,8,9,11,12,14,15,17,18,19,20,21,22,23,24,26,27,29,30,32,33,35,36]

    # data = data[:,mask]
    # data_joint_pos = data[:,7:15]
    # data_joint_vel = data[:,21:]

    # dataRR = data_joint_pos[:,0:2]
    # dataRR_vel = data_joint_vel[:,0:2]
    # dataRL = data_joint_pos[:,2:4]
    # dataRL_vel = data_joint_vel[:,2:4]
    # dataFR = data_joint_pos[:,4:6]
    # dataFR_vel = data_joint_vel[:,4:6]
    # dataFL = data_joint_pos[:,6:]
    # dataFL_vel = data_joint_vel[:,6:]

    # data_joint_pos = np.concatenate((dataFL,dataFR,dataRL,dataRR),axis=-1)
    # data_joint_vel = np.concatenate((dataFL_vel,dataFR_vel,dataRL_vel,dataRR_vel),axis=-1)

    # solo8_data = make_solo8(data, data_joint_pos, data_joint_vel)

    solo8_data = data[:,3:-4] # Ingore xyz position and foot contact

    if chosen_feats is not None:
        solo8_data = solo8_data[:, chosen_feats]
    
    return solo8_data
            


def pad_data(data, num_steps):
    n_pad = num_steps - len(data)%num_steps
    pad_data = np.concatenate([data, np.zeros((n_pad, data.shape[1]))])
    return pad_data


def prepare_data_for_pca(data_dir, num_steps, chosen_feats = None):
    traj_fs = sorted(glob.glob(osp.join(data_dir,"*.csv")))

    # trajs = np.stack([load_traj(f) for f in traj_fs])
    trajs = [load_traj(f, chosen_feats) for f in traj_fs]
    trajs = [pad_data(e, num_steps) for e in trajs]
    
    splitted_trajs = []
    for traj in trajs:
        split_traj = [traj[i:i+num_steps].reshape(-1) for i in range(0, len(traj), num_steps)]
        splitted_trajs.append(split_traj)
    
    splitted_trajs = np.concatenate(splitted_trajs)
    return splitted_trajs

def train_pca(data_dir=DATA_DIR, num_steps=16, n_comp=64, chosen_feats=None):
    data = prepare_data_for_pca(data_dir, num_steps, chosen_feats)
    pca = PCA(n_components=max(16, n_comp))
    pca.fit(data)
    
    joblib.dump(pca, "motion_pca.joblib")


if __name__ == "__main__":
    cfg_f = osp.join(osp.dirname(osp.realpath(__file__)), "cfg.yaml")
    cfg = YAML().load(open(cfg_f, "r"))

    train_pca(num_steps=cfg["pca"]["num_steps"], n_comp=cfg["pca"]["n_components"], chosen_feats=cfg["pca"]["train_feats"])
