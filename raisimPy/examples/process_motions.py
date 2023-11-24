import os
import numpy as np
import raisimpy as raisim
import time

raisim.World.setLicenseFile(os.path.dirname(
    os.path.abspath(__file__)) + "/../../rsc/activation.raisim")
go1_urdf_file = os.path.dirname(os.path.abspath(
    __file__)) + "/../../rsc/go1/urdf/go1_v2.urdf"

world = raisim.World()
world.setTimeStep(0.001)
server = raisim.RaisimServer(world)
ground = world.addGround()

# Control Information
# Frequency: 50Hz


go1 = world.addArticulatedSystem(go1_urdf_file)
go1.setName("Go1")

go1_nominal_joint_config = np.array([0, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8, -1.7,
                                        0.00, 0.8, -1.7, 0.0, 0.8, -1.7, 0.00, 0.8, -1.7])

offset_pl = np.array([0.0,0.8,-1.7])

go1.setGeneralizedCoordinate(go1_nominal_joint_config)
go1.setPdGains(100*np.ones([18]), 10*np.ones([18]))
go1.setPdTarget(go1_nominal_joint_config, np.zeros([18]))

server.launchServer(8080)

def updateTargets(index):
    p_targets = np.zeros(19)
    d_targets = np.zeros(18)
    p_targets[7:10] = dataRR[index] + offset_pl
    p_targets[10:13] = dataRL[index] + offset_pl
    p_targets[13:16] = dataFR[index] + offset_pl
    p_targets[16:] = dataFL[index] + offset_pl

    d_targets[:6] = data_vel_body[index]
    d_targets[6:9] = dataRR_vel[index]
    d_targets[9:12] = dataRL_vel[index]
    d_targets[12:15] = dataFR_vel[index]
    d_targets[15:] = dataFL_vel[index]

    go1.setPdTarget(p_targets,d_targets)
    pos,vel = go1.getState()
    footIndex = go1.getBodyIdx("LF_SHANK")
    cotacts = [0,0,0,0]
    for contact in go1.getContacts():
        if contact.getlocalBodyIndex() == go1.getBodyIdx("RR_calf"):
            cotacts[0] = 1
        elif contact.getlocalBodyIndex() == go1.getBodyIdx("RL_calf"):
            cotacts[1] = 1
        elif contact.getlocalBodyIndex() == go1.getBodyIdx("FR_calf"):
            cotacts[2] = 1
        elif contact.getlocalBodyIndex() == go1.getBodyIdx("FL_calf"):
            cotacts[3] = 1
    state = np.hstack((pos,vel,cotacts))
    world.integrate()
    return state

time.sleep(2)
world.integrate1()



directory='go1_motions'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    data = np.loadtxt(f,delimiter=",")
    data = data[400:-150,:]
    n_data = len(data)

    # data front right --> sim rear left

    data_pos = data[:,6:18]
    data_vel = data[:,18:]
    data_vel_body = data[:,:6]

    dataFL = data_pos[:,0:3]
    dataFL_vel = data_vel[:,0:3]
    dataFR = data_pos[:,3:6]
    dataFR_vel = data_vel[:,3:6]
    dataRL = data_pos[:,6:9]
    dataRL_vel = data_vel[:,6:9]
    dataRR = data_pos[:,9:]
    dataRR_vel = data_vel[:,9:]

    # SIM INFORMATION
    # Front Left --> [16:]
    # Front Right --> [13:16]
    # Rear Left --> [10:13]
    # Rear Right --> [7:10]

    # 2 seconds wait to set default position
    print('processing ' + filename[:-4] + " motion")
    go1.setGeneralizedCoordinate(go1_nominal_joint_config)
    go1.setPdTarget(go1_nominal_joint_config, np.zeros([18]))
    for i in range(2000):
        time.sleep(0.001)
        world.integrate()

    ep_length = len(data)
    counter = 0
    states = []
    for i in range(1000000000):
        time.sleep(0.001)
        world.integrate()
        if i % 20 == 0:
            robot_state = updateTargets(counter)
            states.append(robot_state)
            counter+=1
            if counter >= n_data:
                break
    np.savetxt('go1_motions_w_foot_contact/preprocessed_'+filename, states, delimiter=",")

server.killServer()
