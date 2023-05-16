import numpy as np
from scipy.integrate import solve_ivp

class car:
    """
    Use this class to solve the equation of motion of a simple car. The car has a mass and it encounters friction and
    engine force. The acceleration is similar to opening and closing of a valve. The force from engine is a constant
    and it can either be in the positive or negative or zero. The car is manipulated by opening the valve in the
    positive and negative direction.
    """

    g = 9.81

    def __init__(self, carMass, frictionCoeff, force_scale):
        self.mass = carMass
        self.mu = frictionCoeff
        self.x = []
        self.x.append(0)
        self.force_scale = force_scale
    
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
