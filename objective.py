from constants import TOP_Y


def create_drone_objective_for_mpc(
    hole_center=(0.0, TOP_Y),
    Q_y=1.0,
    Q_x=0.25,
    Q_vel=0.4,
    Q_final=100.0,
):
    """
    Creates an objective function for use with MPCContinuous.

    Returns:
        objective_function: A function that takes a state and returns a cost
    """

    def objective_function(state):
        """Returns a stage cost for the drone MPC problem."""
        x_pos, y_pos, _, vx, vy, _ = (
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5],
        )
        x_h, y_h = hole_center

        # Distance-to-hole cost
        cost = Q_y * (y_h - y_pos) ** 2
        cost += Q_x * (x_h - x_pos) ** 2

        # Velocity penalty
        speed_sq = vx**2 + vy**2
        cost += Q_vel * speed_sq

        return cost

    return objective_function
