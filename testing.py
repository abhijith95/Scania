import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class car:

    g = 9.81

    def __init__(self, carMass, frictionCoeff):
        self.mass = carMass
        self.mu = frictionCoeff
        self.x = []
        self.x.append(0)
    
    def eom(self,t,x,f):
        """
        Function that returns dxdot of the system for the ivp to solve
        :param t:
        :param x: [x1, x2] vector of state variable
        :param f: the force value to be applied to the car
        :return: dxdt of the system in state space format
        """
        x1,x2 = x
        x1dot = x2
        x2dot = (f/self.mass) - (self.mu * self.g * np.sign(x2))
        dxdt = [x1dot, x2dot]
        return dxdt

    def solve_for_dt(self, x0, time_span, f):
        """
        Function that solves the system of equation from time = t to t+dt
        :param x0: [position, velocity] before calling the solve function
        :param time_span: [current time, current_time + dt]
        :param f: Scalar variable that applies force to the system
        :return: [position, velocity] at the next time step
        """
        solution = solve_ivp(fun=self.eom, t_span=time_span, t_eval=[time_span[1]],
                            y0=x0, args=[f])
        return float(solution.y[0]), float(solution.y[1])

    def control_law(self, e, vel, t, dead_band):
        """
        Function that returns the control variable which either takes the value of ON or OFF. The control function will 
        be a sigmoid function where if the output is above 0.5 the controller will be ON otherwise it will be OFF.

        Parameters
        ----------
        e : float
            error in the current time step
        vel : float
            current velocity of the vehicle
        t : float
            current time
        dead_band : list
                    list of size two containing the lower and upper limit of deviation from target position
        """
        
        ke,kv,kt = 0.00, 0.1, 0.0
        x = ke*(abs(e))
        if vel != 0:
            x+= kv/abs(abs(vel)) + kt/t
        # saturate x to +/- 10 because there is no need to go beyond this range
        if x > 20:
            x = 10
        if x < -20:
            x = -10
        try:
            ux = 1/(1+math.exp(-x))
        except:
            ux = 0
        
        if dead_band[0] <= e <= dead_band[1]:
            self.x.append(0)
            return 0
        if ux >= 0.5:
            self.x.append(ux)
            return (1*np.sign(e))
        self.x.append(0)
        return 0

def main():
    c = car(carMass=15, frictionCoeff=0.2)
    t, pos, vel, err = 0, 0, 0, 0
    dt = 0.1
    force_scale = 50
    goal_pos = 50
    dead_band = [-1, 1]

    position, velocity, time, u = [], [], [], []
    position.append(pos)
    velocity.append(vel)
    time.append(t)
    u.append(0)
    while t<=30:
        force = c.control_law(err, vel, t, dead_band)
        u.append(force)
        pos, vel = c.solve_for_dt(x0=[pos, vel], time_span=[t, t+dt], f=force*force_scale)
        t+=dt
        position.append(pos)
        velocity.append(vel)
        time.append(t)
        err = goal_pos - pos

    # find settling time. Setting time is the min time after which the position remains within the deadband
    temp = []
    for i in reversed(range(len(time))):
        if goal_pos+dead_band[0]<= int(position[i]) <= goal_pos + dead_band[1]:
            temp.append(time[i])
        else:
            break
    
    # characterisitc texts
    
    # plotting graphs
    fig, axs = plt.subplots(3)
    axs[0].plot(time, position)
    axs[0].plot(time, [goal_pos]*len(time), 'r--')
    axs[0].plot(time, [goal_pos+dead_band[0]]*len(time), 'g')
    axs[0].plot(time, [goal_pos+dead_band[1]]*len(time), 'g')
    axs[0].plot(time, [max(position)]*len(time), 'k')
    
    if temp:
        axs[0].plot([temp[-1]]*len(range(0, int(max(position))+5)), range(0, int(max(position))+5), "m--")
        txt = "Overshoot : " + str(max(position)) + "\nSettling time: " + str(int(temp[-1])) + " s"
        fig.text(1,1, txt, ha='left', va='top')
        
    axs[0].grid()
    
    axs[0].set_ylabel("Position (m)")
    
    axs[1].plot(time, velocity)
    axs[1].grid()
    axs[1].set_ylabel("Velocity (m/s)")
    axs[2].plot(time, u)
    axs[2].grid()
    axs[2].set_ylabel("Valve position")
    axs[2].set_xlabel("Time (s)")
    
    fig2,axs2 = plt.subplots(2)
    axs2[0].plot(position,c.x)
    axs2[0].grid()
    # axs2[1].plot(velocity,c.x)
    # axs2[1].grid()
    
    plt.show()

if __name__ == "__main__":
    main()