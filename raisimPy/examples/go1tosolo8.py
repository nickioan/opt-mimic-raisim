import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(
    os.path.abspath(__file__)) + "/../../rsc/activation.raisim")

solo8_urdf_file = os.path.dirname(os.path.abspath(
    __file__)) + "/../../rsc/solo8_v7/solo8.urdf"

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()

data = np.loadtxt("go1_motions_processed/preprocessed_bound forward slowly.csv",delimiter=",")
mask = [0,1,2,3,4,5,6,8,9,11,12,14,15,17,18,19,20,21,22,23,24,26,27,29,30,32,33,35,36]
#mapping = [1., 1., 1., 1., -1.,-1., -1., -1.]
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

# data front right --> sim rear left


solo8 = world.addArticulatedSystem(solo8_urdf_file)
solo8.setName("solo8")
solo8_nominal_joint_config = np.zeros(solo8.getGeneralizedCoordinateDim())
solo8_nominal_joint_config[2] = 0.5
solo8_nominal_joint_config[3] = 1
solo8_nominal_joint_config[7:] = np.array([np.pi/4.0, -np.pi/2.0, np.pi/4.0, -np.pi/2.0,
                                           -np.pi/4.0, np.pi/2.0, -np.pi/4.0, np.pi/2.0])
solo8.setGeneralizedCoordinate(solo8_nominal_joint_config)
solo8.setPdGains(3*np.ones([14]), 0.5*np.ones([14]))
solo8.setPdTarget(solo8_nominal_joint_config, np.zeros([14]))


server.launchServer(8080)

ep_length = len(data)
run_time = ep_length*20+1

def updateTargets(index):
    p_targets = np.zeros(15)
    d_targets = np.zeros(14)

    p_targets[:7] = data[index,:7]
    p_targets[7:] = data_joint_pos[index,:]
    d_targets[:6] = data[index,15:21]
    d_targets[6:] = data_joint_vel[index,:]

    solo8.setPdTarget(p_targets,d_targets)
    world.integrate()

time.sleep(2)
world.integrate1()
counter = 1

for i in range(1000000000):
    time.sleep(0.001)
    world.integrate()
    if i % 20 == 0:
        updateTargets(counter)
        counter+=1

server.killServer()
