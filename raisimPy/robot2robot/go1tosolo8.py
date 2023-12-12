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

time.sleep(2)
world.integrate1()

directory='go1_motions'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    data = np.loadtxt(f,delimiter=",")
    data = data[400:-150,:]
    n_data = len(data)

    # Raw 30:
    # 0-5: body vel 6d
    # 6-8: FL pos
    # 9-11: FR pos
    # 12-14: RL pos
    # 15-17: RR pos
    # 18-20: FL vel
    # 21-23: FR vel
    # 24-26: RL vel
    # 27-29: RR vel
    
    # go1 37 (additional 3d body pos + 4d quaternion rotation) w/out contact (4d)
    # 0-2: xyz position
    # 3-6: quaternions
    # 7-9: rear right, joint angle, direction from body to toe
    # 10-12: rear left
    # 13-15: front right
    # 16-18: front left

    # solo8:
    # ...
    # 7-9: FL, joint angle, direction from body to toe
    # 10-12: FR
    # 13-15: RL
    # 16-18: RR
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
            # SET TARGET POSITION/VELOCITIES
            # FOR SOLO8 Don't Use Velocities (Keep them zero)
            p_targets = np.zeros(19)
            d_targets = np.zeros(18)
            p_targets[7:10] = dataRR[counter] + offset_pl
            p_targets[10:13] = dataRL[counter] + offset_pl
            p_targets[13:16] = dataFR[counter] + offset_pl
            p_targets[16:] = dataFL[counter] + offset_pl

            d_targets[:6] = data_vel_body[counter]
            d_targets[6:9] = dataRR_vel[counter]
            d_targets[9:12] = dataRL_vel[counter]
            d_targets[12:15] = dataFR_vel[counter]
            d_targets[15:] = dataFL_vel[counter]

            go1.setPdTarget(p_targets,d_targets)
            world.integrate()
            # READ STATE INFORMATION FROM SIMULATOR
            pos,vel = go1.getState()
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
            
            # Masking should aim for 29, initially 37
            go1_data_sim = np.hstack((pos, vel))
            mask = [0,1,2,3,4,5,6,8,9,11,12,14,15,17,18,19,20,21,22,23,24,26,27,29,30,32,33,35,36]
            solo8_data = go1_data_sim[mask]
            solo8_data_joint_pos = solo8_data[7:15]
            solo8_data_joint_vel = solo8_data[21:]

            solo8_body_vel = solo8_data[15:21]
            solo8_dataRR = solo8_data_joint_pos[0:2]
            solo8_dataRR_vel = solo8_data_joint_vel[0:2]
            solo8_dataRL = solo8_data_joint_pos[2:4]
            solo8_dataRL_vel = solo8_data_joint_vel[2:4]
            solo8_dataFR = solo8_data_joint_pos[4:6]
            solo8_dataFR_vel = solo8_data_joint_vel[4:6]
            solo8_dataFL = solo8_data_joint_pos[6:]
            solo8_dataFL_vel = solo8_data_joint_vel[6:]
            solo8_data_joint_pos = np.concatenate((solo8_dataFL,solo8_dataFR,solo8_dataRL,solo8_dataRR))
            solo8_data_joint_vel = np.concatenate((solo8_dataFL_vel,solo8_dataFR_vel,solo8_dataRL_vel,solo8_dataRR_vel))
            state = np.concatenate((solo8_data[0:7], solo8_data_joint_pos,solo8_body_vel, solo8_data_joint_vel,cotacts))
            states.append(state)
            counter+=1
            if counter >= n_data:
                break
    np.savetxt('solo8_motions/'+filename, states, delimiter=",")

server.killServer()
