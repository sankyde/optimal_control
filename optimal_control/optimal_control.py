import gym
import numpy as np
from scipy import linalg


masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length=0.5
gravity=9.8
a=gravity/(length*(4.0/3 - masspole/(total_mass)))

b= -1/(length*(4.0/3 - masspole/total_mass))

A=np.array([[0,1,0,0],
            [0,0,a,0],
            [0,0,0,1],
            [0,0,a,0]])
B=np.array([[0],[1/total_mass],[0],[b]])

R=np.eye(1,dtype=int)
Q=5*np.eye(4,dtype=int)

def calc_are():
    P=linalg.solve_continuous_are(A,B,Q,R)
    K=np.dot(np.linalg.inv(R),np.dot(B.T,P))

    return K

def apply_state_controller(K,x):

    u=-np.dot(K,x)
    # print("K",K)
    # print("u",u)
    # print("x",x)
    if u>0:
        return 1,u
    else:
        return 0,u

env=gym.make('CartPole-v0')
obs=env.reset()
N=1000
done=False
for i in range(1000):

    env.render()


    action,force = apply_state_controller(calc_are(),obs)


    abs_force= abs(float(np.clip(force,-10,10)))


    env.env.force_mag=abs_force


    obs,reward,done,_=env.step(action)



env.close()

