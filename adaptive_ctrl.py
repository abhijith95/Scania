from car_mechanics import car
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import draw, show
from enum import Enum


class Controller_state(Enum):
    SMART_CONTROLLER, REGULAR_CONTROLLER = range(2)


class controller:
    min_wt = 5
    max_wt = 35
    weight_buckets = np.arange(start=min_wt, stop=max_wt, step=5)
    control_buckets = [0.75, 0.45, 0.5, 0.5, 0.5, 0.5, 0.4]
    learn_parameter = 0
    allowable_ovrshoot = 1.1

    def __init__(self, dead_band):
        self.dead_band = dead_band
        self.reset_controller()

    def reset_controller(self):
        """
        Function to reset the controller before every maneuver.
        """
        self.old_position = 0
        self.saturate_timer = 0
        self.controller_state = Controller_state.SMART_CONTROLLER
        self.learn_parameter = 0

    def set_goal_pos(self, goal_pos):
        """
        Get the end position of the system.

        Parameters
        ----------
        goal_pos : float
            End position of the car
        """
        self.goal_pos = goal_pos
        self.dead_band_percent = [(i+self.goal_pos)/self.goal_pos for i in self.dead_band]
        self.switch_tolerance = [-0.2, 0.2]

    def find_ctrlr_bucket(self, car_wt):
        """
        Function that will return the current vehicle's bucket index. This will then used by the controller to take
        the correct decision

        Parameters
        ----------
        car_wt : float
            Current weight of the vehicle in Kg.

        Returns
        -------
        int
            Index where the weight lies
        """
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
            current_pos (): float
        Returns:
            Bool: whether to switch or not
        """
        if abs(current_pos_percent) < self.dead_band_percent[0]:
            # if the system is saturating then we need to switch to the regular controller
            if self.switch_tolerance[0] <= (current_pos - self.old_position) <= self.switch_tolerance[1]:
                self.saturate_timer += 1
                if self.saturate_timer > 10:
                    self.learn_parameter = current_pos / self.goal_pos
                    return True
                else:
                    return False
            else:
                self.saturate_timer = 0

        elif abs(current_pos_percent) > self.dead_band_percent[1]:
            # if the system has overshot the allowable overshoot then we switch to regular controller
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
            # The smart controller is the controller which learns with every iteration. It will try to control the
            # system without any undershoot or overshoot.
            if abs(current_pos_percent) <= self.control_buckets[index]:
                output = 1 * np.sign(1 - current_pos_percent)
            else:
                output = 0
            if self.switch_controller(current_pos_percent, current_pos):
                self.controller_state = Controller_state.REGULAR_CONTROLLER

        if self.controller_state == Controller_state.REGULAR_CONTROLLER:
            # Regular bang bang controller that switches ON and OFF the acceleration based on the current position
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
        """
        Function that updates the threshold based on the current vehicle weight.
        """
        if self.learn_parameter != 0:
            print(self.control_buckets[self.learn_index], self.learn_parameter)
            self.control_buckets[self.learn_index] = self.control_buckets[self.learn_index] + \
                                                     (1 - self.learn_parameter)
            return True
        return False


def step_response(run_time, dt, car_mass, controller_obj, car_obj):
    """
    Function that will solve the differential equations for a set run time.

    Parameters
    ----------
    run_time : (int) Total run time of the simulation
    dt : (float) Step time for the simulation
    car_mass : (float) Current weight of the vehicle
    controller_obj : (type controller) This is the class object of controller
    car_obj : (type car) This is the class object of the vehicle, that contains the equations of motion

    Returns
    -------
    List containing the output from the simulation run
    """
    t, pos, vel, err = 0, 0, 0, 0
    position, velocity, time, u, error, ctrlr_state = [], [], [], [], [], []
    position.append(pos)
    velocity.append(vel)
    time.append(t)
    u.append(0)
    ctrlr_state.append(controller_obj.controller_state.value)

    while t <= run_time:
        force = controller_obj.control_output(pos, car_mass)
        u.append(force)
        temp_pos, temp_vel = car_obj.solve_for_dt(x0=[pos, vel], time_span=[t, t + dt], f=force * car_obj.force_scale)

        t += dt
        pos = np.copy(temp_pos)
        vel = np.copy(temp_vel)
        position.append(pos)
        velocity.append(vel)
        time.append(t)
        error.append(err)
        ctrlr_state.append(controller_obj.controller_state.value)

    return [position, velocity, time, u, error, ctrlr_state]


def plot_sys_response(axs, plot_data, learning_iteration, plot_window):
    """
    Function that will plot the system response to a step input
    Parameters
    ----------
    plot_window : (int)
    axs : Axes to plot in
    plot_data : A list of input data required for plotting
    learning_iteration : Number of iterations until the system stopped learning
    """
    iteration_time, iteration_position, goal_pos, dead_band_percent, allowable_ovrshoot, plot_names, plot_title = \
        plot_data
    plot_names.insert(0, "Goal pos tolerance")
    plot_names.insert(0, "Goal pos tolerance")
    plot_names.insert(2, "Allowable overshoot")

    plt.figure(plot_window)
    plt.title(plot_title)
    plt.plot(iteration_time[0], [goal_pos * dead_band_percent[0]] * len(iteration_time[0]), 'r--')
    plt.plot(iteration_time[0], [goal_pos * dead_band_percent[1]] * len(iteration_time[0]), 'r--')
    plt.plot(iteration_time[0], [goal_pos * allowable_ovrshoot] * len(iteration_time[0]), 'b--')
    plt.grid()
    plt.ylabel("Position (m)")
    for i in range(learning_iteration):
        plt.plot(iteration_time[i], iteration_position[i])

    plt.legend(plot_names)
    draw()


def main():
    dt = 0.1
    goal_pos = 50
    dead_band = [-0.5, 1]
    ctrl = controller(dead_band)
    ctrl.set_goal_pos(goal_pos)
    car_mass = 10
    run_cases = [[car_mass, 0.15, 50], [car_mass, 0.18, 50], [car_mass, 0.18, 40]]
    fig, axs = plt.subplots(1)

    for i, run_case in enumerate(run_cases):
        learning = True
        learning_iteration = 0
        iteration_position, iteration_time, plot_names = [], [], []
        # plot_colors = ['g--', 'b--', 'm', 'y', 'k']

        while learning_iteration < 7:
            ctrl.reset_controller()
            learning_iteration += 1
            c = car(carMass=run_case[0], frictionCoeff=run_case[1], force_scale=run_case[2])

            plot_names.append(ctrl.control_buckets[ctrl.find_ctrlr_bucket(car_mass)])
            position, velocity, time, u, error, ctrlr_state = step_response(run_time=30, dt=dt, car_mass=car_mass,
                                                                            controller_obj=ctrl, car_obj=c)
            learning = ctrl.learn()
            iteration_position.append(position)
            iteration_time.append(time)

        plot_title = "Car mass: " + str(run_case[0]) + " friction: " + str(run_case[1]) + " force: " + str(run_case[2])
        plot_sys_response(axs=axs, plot_data=[iteration_time, iteration_position, goal_pos,
                                              ctrl.dead_band_percent, ctrl.allowable_ovrshoot, plot_names,
                                              plot_title],
                          learning_iteration=learning_iteration, plot_window=i+1)

    show()


if __name__ == "__main__":
    main()
