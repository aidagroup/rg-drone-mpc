"""
MPC implementation for an underwater drone system using the regelum package.

This implementation includes:
1. A state transition model for the drone dynamics
2. An objective function that guides the drone to a target hole
3. Obstacle avoidance constraints for an elliptical obstacle
4. Custom constraint integration with the MPCContinuous framework
"""

import numpy as np
from regelum import Node, Graph
import casadi as ca
from typing import Callable, Tuple
from regelum.node.core.variable import Variable
from regelum import symbolic_mode
from IPython.display import display
from regelum.node.memory.buffer import DataBuffer
from objective import create_drone_objective_for_mpc
from constraints import add_drone_obstacle_constraints

# Constants
from constants import (
    TIME_STEP_SIZE,
    DRONE_MASS,
    DRONE_INERTIA,
    DRAG_COEFF,
    OFFSET_LAT,
    GRAVITY,
    MAX_F_LONG,
    MAX_F_LAT,
    TOP_Y,
)

from visualization import animate_drone_trajectory
from system import UnderwaterDrone
from regelum.node.classic_control.controllers.mpc import MPCContinuous


class MPCContinuousWithConstraints(MPCContinuous):
    """MPCContinuous controller with support for custom constraints."""

    def __init__(
        self,
        controlled_system: Node,
        controlled_state: Variable,
        control_dimension: int,
        objective_function: Callable[[np.ndarray], float],
        control_bounds: Tuple[np.ndarray, np.ndarray],
        step_size: float = 0.01,
        prediction_horizon: int = 3,
        prediction_method: MPCContinuous.PredictionMethod = MPCContinuous.PredictionMethod.RK4,
        name: str = "mpc",
        custom_constraints: Callable[[ca.Opti, ca.MX, ca.MX], None] = None,
    ):
        """Initialize MPC controller with custom constraints.

        Args:
            controlled_system: Controlled system.
            controlled_state: Controlled state.
            control_dimension: Control dimension.
            objective_function: Objective function.
            control_bounds: Control bounds.
            step_size: Step size.
            prediction_horizon: Prediction horizon.
            prediction_method: Integration method for prediction (RK4 or Euler).
            name: Name of the MPC controller.
            custom_constraints: Optional function to add custom constraints to the optimization problem.
                The function should take (opti, X, U) where X is the state matrix of shape (state_dim, N+1)
                and U is the control matrix of shape (control_dim, N).
        """
        # Store custom constraints before parent initialization
        self.custom_constraints = custom_constraints

        # Initialize parent class
        super().__init__(
            controlled_system=controlled_system,
            controlled_state=controlled_state,
            control_dimension=control_dimension,
            objective_function=objective_function,
            control_bounds=control_bounds,
            step_size=step_size,
            prediction_horizon=prediction_horizon,
            prediction_method=prediction_method,
            name=name,
        )

        # Maintain a reference to last successful control for error handling
        self.last_successful_control = np.zeros(control_dimension)

    def _create_optimization_problem(self) -> Callable[[np.ndarray], np.ndarray]:
        """Override the parent class method to add custom constraints."""
        state_shape = self.controlled_state.metadata.get("shape")
        state_dim = int(state_shape[0])

        dt = self.step_size
        N = self.prediction_horizon

        opti = ca.Opti()
        X = opti.variable(state_dim, N + 1)
        U = opti.variable(self.control_dimension, N)

        cost = 0
        for k in range(N):
            cost += self.objective_function(X[:, k])

        cost += self.objective_function(X[:, N])

        opti.minimize(cost)

        x0 = opti.parameter(state_dim)
        opti.subject_to(X[:, 0] == x0)

        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]

            if self.prediction_method == self.PredictionMethod.RK4:
                with symbolic_mode():
                    k1 = self.controlled_system.state_transition_map(x_k, u_k)
                    k2 = self.controlled_system.state_transition_map(
                        x_k + dt / 2 * k1, u_k
                    )
                    k3 = self.controlled_system.state_transition_map(
                        x_k + dt / 2 * k2, u_k
                    )
                    k4 = self.controlled_system.state_transition_map(x_k + dt * k3, u_k)
                x_next = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            else:  # Euler
                with symbolic_mode():
                    dx = self.controlled_system.state_transition_map(x_k, u_k)
                x_next = x_k + dt * dx

            opti.subject_to(X[:, k + 1] == x_next)

        u_min = self.control_bounds[0][:, None]
        u_max = self.control_bounds[1][:, None]
        opti.subject_to(ca.vec(U) >= ca.vec(ca.DM(np.repeat(u_min, N, axis=1))))
        opti.subject_to(ca.vec(U) <= ca.vec(ca.DM(np.repeat(u_max, N, axis=1))))

        # Add custom constraints if provided
        if self.custom_constraints is not None:
            self.custom_constraints(opti, X, U)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        self.opti = opti
        self.x0 = x0
        self.U = U
        self.X = X  # Store X for external access

        return self.solve_mpc

    def solve_mpc(self, current_state: np.ndarray) -> np.ndarray:
        """Solve the MPC optimization problem with error handling.

        Args:
            current_state: Current state of the system

        Returns:
            Optimal control input for the current state
        """
        self.opti.set_value(self.x0, current_state)

        try:
            sol = self.opti.solve()
            # Store successful control for fallback
            self.last_successful_control = sol.value(self.U[:, 0])
            return self.last_successful_control
        except RuntimeError as e:
            # Handle solver failures gracefully
            print(f"Warning: MPC solver failed: {e}")
            print(
                "Returning last successful control. Consider adjusting model parameters or constraints."
            )
            return self.last_successful_control


class UnderwaterDroneNode(Node):
    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        v_x=0.0,
        v_y=0.0,
        omega=0.0,
    ):
        super().__init__(
            name="drone",
            inputs=[
                "mpc_1.mpc_action"
            ],  # Updated from mpc_1.action to point directly to mpc controller
            step_size=TIME_STEP_SIZE,
            is_root=True,
            is_continuous=True,
        )
        self.drone = UnderwaterDrone(
            x=x,
            y=y,
            theta=theta,
            v_x=v_x,
            v_y=v_y,
            omega=omega,
        )
        self.state = self.define_variable(
            "state",
            shape=(6,),
            value=np.array([x, y, theta, v_x, v_y, omega]),
            metadata={"units": "m, m, rad, m/s, m/s, rad/s", "shape": (6,)},
        )

    def state_transition_map(self, x, u):
        """
        Define the drone dynamics for the MPC prediction model.

        Args:
            x: State vector [x, y, theta, vx, vy, omega]
            u: Control vector [F_long, F_lat]

        Returns:
            dx: State derivative vector
        """
        # Extract state components
        _, _, Th, Vx, Vy, Om = x[0], x[1], x[2], x[3], x[4], x[5]
        Fl, Ft = u[0], u[1]

        # Drone parameters
        m = DRONE_MASS
        I = DRONE_INERTIA
        Cd = DRAG_COEFF
        offset_lateral = OFFSET_LAT
        gravity = GRAVITY

        # Compute forces in inertial frame
        c = ca.cos(Th)
        s = ca.sin(Th)
        Fx_inertial = Fl * c - Ft * s
        Fy_inertial = Fl * s + Ft * c

        # Drag forces - improved numerical stability
        speed_sq = Vx * Vx + Vy * Vy
        speed = ca.sqrt(speed_sq + 1e-6)

        drag_dir_x = -Vx / (speed + 1e-4)
        drag_dir_y = -Vy / (speed + 1e-4)

        F_drag_x = Cd * speed * speed * drag_dir_x
        F_drag_y = Cd * speed * speed * drag_dir_y

        # Net forces
        Fx_net = Fx_inertial + F_drag_x
        Fy_net = Fy_inertial + F_drag_y - gravity

        # Torque
        tau = offset_lateral * Ft

        # State derivatives
        dx_pos = Vx
        dy_pos = Vy
        dtheta = Om
        dvx = Fx_net / m
        dvy = Fy_net / m
        domega = tau / I

        # Return state derivative vector
        return ca.vertcat(dx_pos, dy_pos, dtheta, dvx, dvy, domega)

    def step(self):
        action = self.resolved_inputs.find("mpc_1.mpc_action")
        self.drone.step(action.value)
        self.state.value = np.array(self.drone.state())


if __name__ == "__main__":
    # Create the drone node
    drone_node = UnderwaterDroneNode()

    # Create objective function
    obj_func = create_drone_objective_for_mpc(
        hole_center=(0.0, TOP_Y), Q_y=1.0, Q_x=0.25, Q_vel=0.4, Q_final=100.0
    )

    # Set control bounds
    u_min = np.array([-MAX_F_LONG, -MAX_F_LAT])
    u_max = np.array([MAX_F_LONG, MAX_F_LAT])
    control_bounds = (u_min, u_max)

    # Create the MPC controller directly with obstacle avoidance constraints
    mpc_controller = MPCContinuousWithConstraints(
        controlled_system=drone_node,
        controlled_state=drone_node.state,
        control_dimension=2,  # Longitudinal and lateral force
        objective_function=obj_func,
        control_bounds=control_bounds,
        step_size=TIME_STEP_SIZE,
        prediction_horizon=15,
        prediction_method=MPCContinuous.PredictionMethod.RK4,
        name="mpc",  # Note: changed from mpc_1 to mpc
        custom_constraints=add_drone_obstacle_constraints,
    )

    # Create data buffer for logging
    data_buffer = DataBuffer(
        variable_full_names=["drone_1.state", "mpc_1.mpc_action"],  # Updated node names
        step_sizes=[TIME_STEP_SIZE, TIME_STEP_SIZE],
        buffer_sizes=[400, 400],
    )

    # Create and run the graph
    graph = Graph(
        [drone_node, mpc_controller, data_buffer],
        debug=True,
        initialize_inner_time=True,
        states_to_log=["drone_1.state", "mpc_1.mpc_action"],  # Updated node names
    )
    graph.resolve(graph.variables)

    # Simulation loop
    for i in range(400):
        graph.step()

    # Display the animation with actions and save it to a file
    print("Creating animation of the drone MPC simulation...")

    # Extract data from buffer
    states = data_buffer.find_variable(
        "buffer[drone_1.state]"
    ).value  # Updated node name
    actions = data_buffer.find_variable(
        "buffer[mpc_1.mpc_action]"
    ).value  # Updated node name

    # Display the animation
    visualization_result = animate_drone_trajectory(states, actions)
    display(visualization_result)
