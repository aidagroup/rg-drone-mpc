from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from constants import (
    TIME_STEP_SIZE,
    DRONE_RADIUS,
    TOP_Y,
    HOLE_WIDTH,
)
import numpy as np


def animate_drone_trajectory(
    states, actions=None, save_animation=False, filename="drone_mpc_simulation.mp4"
):
    """
    Create an animation of the drone's trajectory.

    Args:
        states: Buffer containing drone states [x, y, theta, vx, vy, omega]
        actions: Optional buffer containing control actions [F_long, F_lat]
        save_animation: Whether to save the animation to a file
        filename: Filename to use when saving the animation

    Returns:
        Animation object
    """
    # Extract state data - handle different possible formats
    if hasattr(states, "buffer"):
        # DataBuffer variable object
        states_data = states.buffer
    elif hasattr(states, "value") and hasattr(states.value, "buffer"):
        # DataBuffer object inside a variable
        states_data = states.value.buffer
    else:
        # Assume numpy array
        states_data = states

    # Extract action data if provided
    actions_data = None
    if actions is not None:
        if hasattr(actions, "buffer"):
            actions_data = actions.buffer
        elif hasattr(actions, "value") and hasattr(actions.value, "buffer"):
            actions_data = actions.value.buffer
        else:
            actions_data = actions

    # Create figure with adjusted size to accommodate square main plot
    fig = plt.figure(figsize=(16, 10))

    # Main trajectory plot - now with more width to accommodate square aspect ratio
    ax_traj = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)

    # Set limits for the tank
    tank_left = -2.5
    tank_right = 2.5
    tank_bottom = 0
    tank_top = TOP_Y

    # Calculate plot limits with some margin
    x_margin = 0.5
    y_margin = 0.5
    x_min = tank_left - x_margin
    x_max = tank_right + x_margin
    y_min = tank_bottom - y_margin
    y_max = tank_top + y_margin

    # Set the axis limits
    ax_traj.set_xlim(x_min, x_max)
    ax_traj.set_ylim(y_min, y_max)

    # Set equal aspect ratio to ensure square plot
    ax_traj.set_aspect("equal")

    ax_traj.set_xlabel("X Position (m)")
    ax_traj.set_ylabel("Y Position (m)")
    ax_traj.set_title("Underwater Drone MPC Simulation")
    ax_traj.grid(True)

    # Velocity plot
    ax_vel = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Velocity (m/s)")
    ax_vel.set_title("Drone Velocities")
    ax_vel.grid(True)

    # Control inputs plot
    ax_ctrl = plt.subplot2grid((3, 4), (0, 2), rowspan=3, colspan=2)
    ax_ctrl.set_xlabel("Time (s)")
    ax_ctrl.set_ylabel("Control Force (N)")
    ax_ctrl.set_title("MPC Control Actions")
    ax_ctrl.grid(True)

    # Draw tank walls
    ax_traj.plot(
        [tank_left, tank_left],
        [tank_bottom, tank_top],
        "k-",
        linewidth=2,
        label="Tank Wall",
    )
    ax_traj.plot([tank_right, tank_right], [tank_bottom, tank_top], "k-", linewidth=2)
    ax_traj.plot([tank_left, tank_right], [tank_bottom, tank_bottom], "k-", linewidth=2)

    # Draw the hole at the top
    hole_half_width = HOLE_WIDTH / 2
    ax_traj.plot([tank_left, -hole_half_width], [tank_top, tank_top], "k-", linewidth=2)
    ax_traj.plot([hole_half_width, tank_right], [tank_top, tank_top], "k-", linewidth=2)

    # Mark the hole with a different color
    ax_traj.plot(
        [-hole_half_width, hole_half_width],
        [tank_top, tank_top],
        "g-",
        linewidth=3,
        label="Target Hole",
    )

    # Draw the obstacle (ellipse)
    obstacle = Ellipse(
        (0, 1.1),  # Center coordinates
        width=2 * 0.5,  # Semi-major axis * 2
        height=2 * 0.1,  # Semi-minor axis * 2
        angle=0,  # Orientation angle
        facecolor="red",
        alpha=0.3,
        edgecolor="red",
        linewidth=2,
        label="Obstacle",
    )
    ax_traj.add_patch(obstacle)

    # Create a drone object
    drone_body = plt.Circle(
        (0, 0), radius=DRONE_RADIUS, fc="darkgreen", zorder=3, label="Drone"
    )
    ax_traj.add_patch(drone_body)

    # Create drone nose (direction indicator)
    nose_len = 0.3
    nose_half_w = 0.2
    nose_local_coords = np.array(
        [[nose_len, 0.0], [0.0, nose_half_w], [0.0, -nose_half_w]]
    )
    drone_nose = plt.Polygon(nose_local_coords, fc="darkred", zorder=2)
    ax_traj.add_patch(drone_nose)

    # Add trajectory line
    (trajectory,) = ax_traj.plot(
        [], [], "b-", alpha=0.7, linewidth=1.5, label="Trajectory"
    )

    # Add time text
    time_text = ax_traj.text(0.02, 0.95, "", transform=ax_traj.transAxes)

    # Add legend
    ax_traj.legend(loc="upper right")

    # Determine the animation interval and number of steps
    time_step_size = TIME_STEP_SIZE
    num_steps = len(states_data)

    # Create subsample for smoother animation but complete trajectory
    max_frames = min(200, num_steps)  # Limit to 200 frames for smooth animation
    step_size = max(1, num_steps // max_frames)
    frame_indices = range(0, num_steps, step_size)

    # Prepare the trajectory points (all points for full trajectory)
    full_trajectory_x = [states_data[i][0] for i in range(num_steps)]
    full_trajectory_y = [states_data[i][1] for i in range(num_steps)]

    # Prepare time series data for velocities and controls
    times = np.arange(num_steps) * time_step_size
    vx_data = [states_data[i][3] for i in range(num_steps)]
    vy_data = [states_data[i][4] for i in range(num_steps)]

    # Velocity plot lines
    (vx_line,) = ax_vel.plot([], [], "r-", label="Vx")
    (vy_line,) = ax_vel.plot([], [], "b-", label="Vy")
    ax_vel.legend()

    # Control plot lines
    if actions_data is not None:
        f_long_data = [
            actions_data[i][0] for i in range(min(len(actions_data), num_steps))
        ]
        f_lat_data = [
            actions_data[i][1] for i in range(min(len(actions_data), num_steps))
        ]
        (f_long_line,) = ax_ctrl.plot([], [], "g-", label="F_long")
        (f_lat_line,) = ax_ctrl.plot([], [], "m-", label="F_lat")
        ax_ctrl.legend()

    # Set y limits for the plots based on data
    vx_max = max(abs(np.max(vx_data)), abs(np.min(vx_data))) * 1.2
    vy_max = max(abs(np.max(vy_data)), abs(np.min(vy_data))) * 1.2
    vel_max = max(vx_max, vy_max)
    ax_vel.set_ylim(-vel_max, vel_max)

    if actions_data is not None:
        f_max = (
            max(
                max(abs(np.max(f_long_data)), abs(np.min(f_long_data))),
                max(abs(np.max(f_lat_data)), abs(np.min(f_lat_data))),
            )
            * 1.2
        )
        ax_ctrl.set_ylim(-f_max, f_max)

    # Both velocity and control plots should have the same x-axis range
    ax_vel.set_xlim(0, times[-1])
    ax_ctrl.set_xlim(0, times[-1])

    # Add initial state marker and goal state marker
    ax_traj.plot(
        states_data[0][0], states_data[0][1], "bo", markersize=8, label="Start"
    )
    ax_traj.plot(0, TOP_Y, "go", markersize=8, label="Goal")

    # Create vertical time indicator lines for velocity and control plots
    (time_indicator_vel,) = ax_vel.plot([0, 0], [-vel_max, vel_max], "k--", alpha=0.5)
    if actions_data is not None:
        (time_indicator_ctrl,) = ax_ctrl.plot([0, 0], [-f_max, f_max], "k--", alpha=0.5)

    # Adjust layout to accommodate square aspect ratio
    plt.tight_layout()

    # Further adjust positions to avoid overlap due to aspect ratio
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def init():
        """Initialize the animation"""
        drone_body.center = (states_data[0][0], states_data[0][1])
        drone_nose.set_xy(nose_local_coords)
        trajectory.set_data([], [])
        time_text.set_text("")
        vx_line.set_data([], [])
        vy_line.set_data([], [])
        time_indicator_vel.set_data([0, 0], [-vel_max, vel_max])

        if actions_data is not None:
            f_long_line.set_data([], [])
            f_lat_line.set_data([], [])
            time_indicator_ctrl.set_data([0, 0], [-f_max, f_max])
            return (
                drone_body,
                drone_nose,
                trajectory,
                time_text,
                vx_line,
                vy_line,
                time_indicator_vel,
                f_long_line,
                f_lat_line,
                time_indicator_ctrl,
            )
        else:
            return (
                drone_body,
                drone_nose,
                trajectory,
                time_text,
                vx_line,
                vy_line,
                time_indicator_vel,
            )

    def update(frame_idx):
        """Update the animation for each frame"""
        # Convert frame index to data index
        i = frame_idx * step_size
        if i >= num_steps:
            i = num_steps - 1

        current_time = i * time_step_size

        # Get state for the current frame
        state = states_data[i]
        x, y, theta = state[0], state[1], state[2]

        # Update drone body position
        drone_body.center = (x, y)

        # Update drone nose (direction indicator)
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        nose_world = (R @ nose_local_coords.T).T + np.array([x, y])
        drone_nose.set_xy(nose_world)

        # Update trajectory to show all points up to current frame
        trajectory.set_data(full_trajectory_x[: i + 1], full_trajectory_y[: i + 1])

        # Update time text
        time_text.set_text(f"Time: {current_time:.2f}s")

        # Update velocity plot
        vx_line.set_data(times[: i + 1], vx_data[: i + 1])
        vy_line.set_data(times[: i + 1], vy_data[: i + 1])
        time_indicator_vel.set_data([current_time, current_time], [-vel_max, vel_max])

        # Update control plot if actions are available
        if actions_data is not None:
            idx = min(i, len(actions_data) - 1)
            f_long_line.set_data(times[: idx + 1], f_long_data[: idx + 1])
            f_lat_line.set_data(times[: idx + 1], f_lat_data[: idx + 1])
            time_indicator_ctrl.set_data([current_time, current_time], [-f_max, f_max])

            return (
                drone_body,
                drone_nose,
                trajectory,
                time_text,
                vx_line,
                vy_line,
                time_indicator_vel,
                f_long_line,
                f_lat_line,
                time_indicator_ctrl,
            )
        else:
            return (
                drone_body,
                drone_nose,
                trajectory,
                time_text,
                vx_line,
                vy_line,
                time_indicator_vel,
            )

    # Create animation
    interval = 50  # ms, fixed for smoother playback
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        init_func=init,
        interval=interval,
        blit=True,
        repeat=True,
    )

    # Save the animation to a file if requested
    if save_animation:
        try:
            print(f"Saving animation to {filename}...")
            # Try different writers until one works
            writers = ["ffmpeg", "imagemagick", "pillow"]
            saved = False

            for writer in writers:
                try:
                    print(f"Attempting to save with {writer} writer...")
                    anim.save(filename, writer=writer)
                    print(
                        f"Animation saved successfully to {filename} using {writer} writer"
                    )
                    saved = True
                    break
                except Exception as e:
                    print(f"Error saving with {writer}: {e}")

            if not saved:
                print(
                    "All animation saving attempts failed. Please ensure you have a video writer installed."
                )
                print("You can install ffmpeg with: sudo apt-get install ffmpeg")
        except Exception as e:
            print(f"Unexpected error in animation saving: {e}")

    plt.close()  # Close the figure to prevent it from displaying twice
    return HTML(anim.to_jshtml())  # Return animation as HTML for display in notebooks
