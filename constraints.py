def add_drone_obstacle_constraints(opti, X, U):
    """
    Add obstacle avoidance constraints to the drone MPC problem.

    Args:
        opti: CasADi optimizer
        X: State variables matrix (state_dim, N+1)
        U: Control variables matrix (control_dim, N)
    """
    # Define obstacle parameters
    xc_obs = 0.0
    yc_obs = 1.1
    a_obs = 0.5  # Semi-major axis
    b_obs = 0.1  # Semi-minor axis

    # Safety margin to avoid getting too close to the constraint boundary
    safety_margin = 0.02

    # Define obstacle constraint function with improved numerical stability
    def g_func(x_, y_):
        dx = x_ - xc_obs
        dy = y_ - yc_obs
        # Use a more numerically stable formulation that avoids potential scaling issues
        return ((dx / a_obs) ** 2 + (dy / b_obs) ** 2) - (1.0 + safety_margin)

    # Add constraints for each step in the prediction horizon
    N = X.shape[1] - 1  # Number of prediction steps

    for k in range(1, N + 1):  # Skip the initial state (k=0)
        x_pos = X[0, k]  # x position
        y_pos = X[1, k]  # y position
        gval = g_func(x_pos, y_pos)
        opti.subject_to(gval >= 0.0)  # For ellipse obstacle
