from testing import car
import numpy as np
import math
import matplotlib.pyplot as plt
from enum import Enum


class Controller_state(Enum):
    SMART_CONTROLLER, REGULAR_CONTROLLER = range(2)


class controller:
    min_wt = 5
    max_wt = 35
    weight_buckets = np.arange(start=min_wt, stop=max_wt, step=5)
    control_buckets = [0.75, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4]
    learn_parameter = 0
    allowable_ovrshoot = 1.1

    def __init__(self, dead_band):
        self.dead_band = dead_band
        self.reset_controller()

    def reset_controller(self):
        self.old_position = 0
        self.saturate_timer = 0
        self.controller_state = Controller_state.SMART_CONTROLLER
        self.learn_parameter = 0

    def set_goal_pos(self, goal_pos):
        self.goal_pos = goal_pos
        self.dead_band_percent = [(i+self.goal_pos)/self.goal_pos for i in self.dead_band]
        self.switch_tolerance = [-0.2, 0.2]

    def find_ctrlr_bucket(self, car_wt):
        if car_wt < self.weight_buckets[0]:
            index = 0
        elif car_wt > self.weight_buckets[-1]:
            index = len(self.weight_buckets)
        else:
            for i in range(len(self.weight_buckets)-1):
                if self.weight_buckets[i] <= car_wt < self.weight_buckets[i+1]:
                    index = i
        self.learn_index = index
        return index

    def switch_controller(self, current_pos_percent, current_pos):
        """
        If the system remains outside the tolerance zone, without any change in the position value then this function
        shall trigger a change from smart controller to a regular controller.
        Args:
            current_pos_percent (): float
        Returns:
            Bool: whether to switch or not
        """
        if abs(current_pos_percent) < self.dead_band_percent[0] or \
            abs(current_pos_percent) > self.dead_band_percent[1]:
            # if the system is saturating outside the tolerance zone then we switch to the regular controller
            if self.switch_tolerance[0] <= (current_pos - self.old_position) <= self.switch_tolerance[1]:
                self.saturate_timer += 1
                if self.saturate_timer > 10:
                    self.learn_parameter = current_pos / self.goal_pos
                    return True
                else:
                    return False
            else:
                self.saturate_timer = 0

        elif abs(current_pos_percent) > self.allowable_ovrshoot:
            # if the system has overshot the allowable overshoot then we switch to regular controller
            self.learn_parameter = np.copy(self.allowable_ovrshoot)
            return True

        else:
            # here the system is between the lower band of the tolerance and the allowable overshoot, then we don't
            # transition to the regular controller we stay on the smart controller
            self.saturate_timer = 0
            return False

    def control_output(self, current_pos, car_wt):
        current_pos_percent = current_pos / self.goal_pos
        index = self.find_ctrlr_bucket(car_wt)

        if self.controller_state == Controller_state.SMART_CONTROLLER:
            if abs(current_pos_percent) <= self.control_buckets[index]:
                output = 1 * np.sign(1 - current_pos_percent)
            else:
                output = 0
            if self.switch_controller(current_pos_percent, current_pos):
                self.controller_state = Controller_state.REGULAR_CONTROLLER

        if self.controller_state == Controller_state.REGULAR_CONTROLLER:
            if abs(current_pos_percent) < self.dead_band_percent[0] or \
                    abs(current_pos_percent) > self.dead_band_percent[1]:
                output = 1 * np.sign(1 - current_pos_percent)
                if self.learn_parameter == 0:
                    # if the learn parameter is already set by the controller state transition then we use that instead
                    # of overshoot. Because when switching to regular controller there will be an overshoot, which
                    # will only make the system overshoot more in the initial phase.
                    self.learn_parameter = max(self.learn_parameter, current_pos_percent)

        if self.dead_band_percent[0] <= abs(current_pos_percent) <= self.dead_band_percent[1]:
            output = 0

        self.old_position = np.copy(current_pos)
        return output

    def learn(self):
        if self.learn_parameter != 0:
            print(self.control_buckets[self.learn_index], self.learn_parameter)
            self.control_buckets[self.learn_index] = self.control_buckets[self.learn_index] + \
                                                     ((1 - self.learn_parameter)/2.5)
            # if self.learn_parameter > self.control_buckets[self.learn_index]:
            #     self.control_buckets[self.learn_index] = self.control_buckets[self.learn_index] - 0.05
            # else:
            #     self.control_buckets[self.learn_index] = self.control_buckets[self.learn_index] + 0.05


def main():
    dt = 0.1
    force_scale = 50
    goal_pos = 50
    dead_band = [-0.5, 1]
    ctrl = controller(dead_band)
    ctrl.set_goal_pos(goal_pos)
    iteration = 0
    fig, axs = plt.subplots(3)
    iteration_position = []
    iteration_time = []
    # plot_colors = ['g--', 'b--', 'm', 'y', 'k']

    while iteration < 8:
        ctrl.reset_controller()
        iteration += 1
        car_mass = 10
        c = car(carMass=car_mass, frictionCoeff=0.15)
        t, pos, vel, err = 0, 0, 0, 0
        prev_err = 0
        esum, edot = 0, 0
        position, velocity, time, u, error, ctrlr_state = [], [], [], [], [], []
        error_velocity = []
        position.append(pos)
        velocity.append(vel)
        time.append(t)
        u.append(0)
        ctrlr_state.append(ctrl.controller_state.value)
        error.append(goal_pos - pos)

        while t <= 30:

            force = ctrl.control_output(pos, car_mass)
            u.append(force)
            temp_pos, temp_vel = c.solve_for_dt(x0=[pos, vel], time_span=[t, t+dt], f=force*force_scale)

            t += dt
            pos = np.copy(temp_pos)
            vel = np.copy(temp_vel)
            err = goal_pos - pos
            err_vel = (err - prev_err)/dt
            esum += err*dt
            prev_err = np.copy(err)

            position.append(pos)
            velocity.append(vel)
            time.append(t)
            error.append(err)
            error_velocity.append(err_vel)
            ctrlr_state.append(ctrl.controller_state.value)

        ctrl.learn()
        iteration_position.append(position)
        iteration_time.append(time)
        # plotting the system variable

    axs[0].plot(iteration_time[0], [goal_pos*ctrl.dead_band_percent[0]] * len(iteration_time[0]), 'r--')
    axs[0].plot(iteration_time[0], [goal_pos*ctrl.dead_band_percent[1]] * len(iteration_time[0]), 'r--')
    axs[0].plot(iteration_time[0], [goal_pos*ctrl.allowable_ovrshoot] * len(iteration_time[0]), 'b--')
    axs[0].grid()
    axs[0].set_ylabel("Position (m)")
    for i in range(iteration):
        axs[0].plot(iteration_time[i], iteration_position[i])
    # axs[1].plot(time, u)
    # axs[1].grid()
    # axs[1].set_ylabel("Valve position")
    # axs[2].plot(time, ctrlr_state)
    # axs[2].grid()
    # axs[2].set_ylabel("Controller state")
    # axs[2].set_xlabel("Time (s)")
    plt.show()


if __name__ == "__main__":
    main()
