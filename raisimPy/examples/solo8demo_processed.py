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

filename = "trot forward fast.csv"
data = np.loadtxt("solo8_motions/preprocessed_"+filename,delimiter=",")

solo8 = world.addArticulatedSystem(solo8_urdf_file)
solo8.setName("solo8")

solo8_nominal_joint_config = np.zeros(solo8.getGeneralizedCoordinateDim())
solo8_nominal_joint_config[2] = 0.5
solo8_nominal_joint_config[3] = 1
solo8_nominal_joint_config[7:] = np.array([np.pi/4.0, -np.pi/2.0, np.pi/4.0, -np.pi/2.0,
                                           -np.pi/4.0, np.pi/2.0, -np.pi/4.0, np.pi/2.0])
solo8.setGeneralizedCoordinate(solo8_nominal_joint_config)
kp = 9.8 * 0.75
kd = kp * 0.2
solo8.setPdGains(kp*np.ones([14]), kd*np.ones([14]))
solo8.setPdTarget(solo8_nominal_joint_config, np.zeros([14]))


server.launchServer(8080)

ep_length = len(data)
run_time = ep_length*20+1

def updateTargets(index):
    p_targets = np.zeros(15)
    d_targets = np.zeros(14)

    p_targets = data[index,:15]
    d_targets = data[index,15:]

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
        if counter >= len(data):
            counter = 0

server.killServer()
