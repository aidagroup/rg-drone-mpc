from matplotlib.patches import Ellipse, Rectangle, Polygon
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
import matplotlib.patches as patches


# Define tank constants if not already in constants.py
TANK_WIDTH = 5.0  # Same as tank_right - tank_left


class Tank:
    """
    Manages the container geometry (bounding walls + internal obstacles + top hole),
    handles collisions, and provides a draw() method for visualization.
    Also has methods to init and update water lines that drift left->right.
    """

    def __init__(self, drone_radius=DRONE_RADIUS, water_drift_speed=0.5):
        """
        water_drift_speed: how many x-units per "unit time" the dashes drift.
        """

        # Tank bounding box
        self.x_min = -TANK_WIDTH / 2.0
        self.x_max = TANK_WIDTH / 2.0
        self.y_min = 0.0
        self.y_max = TOP_Y

        # Make a hole at the top, in the center, sized ~ 2 * drone diameter
        self.hole_width = 4.0 * drone_radius

        # Define internal obstacles: "wall with downward horns"
        self.obstacles = [
            {
                "type": "rect",
                "x_min": -0.6,
                "x_max": 0.6,
                "y_min": 2.0,
                "y_max": 2.2,
            },  #  the "bar"
            {
                "type": "rect",
                "x_min": -0.6,
                "x_max": -0.4,
                "y_min": 1.5,
                "y_max": 2.2,
            },  # left "horn"
            {
                "type": "rect",
                "x_min": 0.4,
                "x_max": 0.6,
                "y_min": 1.5,
                "y_max": 2.2,
            },  # right "horn"
        ]

        # For plotting "water lines"
        np.random.seed(0)
        y_lines = np.linspace(self.y_min + 0.2, self.y_max - 0.2, 10)
        y_lines += (np.random.rand(len(y_lines)) - 0.5) * 0.2  # small shuffle

        # Store base segments so we can shift them each frame
        self.water_segments = []
        for y_line in y_lines:
            n_segments = 5  # number of horizontal dashes at this y
            segs = []
            for _ in range(n_segments):
                seg_len = np.random.uniform(0.1, 0.3)
                x_start = np.random.uniform(
                    self.x_min + 0.2, self.x_max - 0.2 - seg_len
                )
                segs.append((x_start, x_start + seg_len))
            self.water_segments.append((y_line, segs))

        self.water_drift_speed = water_drift_speed

        # Create line objects in init_water_visual()
        self.water_line_objects = []  # list of line2D objects
        self._base_segments = None  # to store the original data for each line
        self._cumulative_shift = 0.0  # Track accumulated shift

    def draw_static(self, ax):
        """
        Draw bounding region in light blue, black walls, obstacles.
        Does NOT draw the 'floating' water lines.
        We'll handle them in init/update_water_visual() so we can animate them.
        """
        # Light-blue rectangle
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        water_patch = patches.Rectangle(
            (self.x_min, self.y_min),
            width,
            height,
            facecolor="lightblue",
            edgecolor=None,
            alpha=0.4,
            zorder=0,
        )
        ax.add_patch(water_patch)

        # Walls
        wall_thickness = 0.05

        # Bottom
        bottom_wall = patches.Rectangle(
            (self.x_min, self.y_min), width, wall_thickness, facecolor="black"
        )
        ax.add_patch(bottom_wall)
        # Left
        left_wall = patches.Rectangle(
            (self.x_min, self.y_min), wall_thickness, height, facecolor="black"
        )
        ax.add_patch(left_wall)
        # Right
        right_wall = patches.Rectangle(
            (self.x_max - wall_thickness, self.y_min),
            wall_thickness,
            height,
            facecolor="black",
        )
        ax.add_patch(right_wall)
        # Top left
        hole_left = -self.hole_width / 2
        top_left_width = hole_left - self.x_min
        if top_left_width > 0:
            top_left_wall = patches.Rectangle(
                (self.x_min, self.y_max),
                top_left_width,
                wall_thickness,
                facecolor="black",
            )
            ax.add_patch(top_left_wall)
        # Top right
        hole_right = self.hole_width / 2
        top_right_width = self.x_max - hole_right
        if top_right_width > 0:
            top_right_wall = patches.Rectangle(
                (hole_right, self.y_max),
                top_right_width,
                wall_thickness,
                facecolor="black",
            )
            ax.add_patch(top_right_wall)

        # Obstacles
        for obs in self.obstacles:
            if obs["type"] == "rect":
                w = obs["x_max"] - obs["x_min"]
                h = obs["y_max"] - obs["y_min"]
                rpatch = patches.Rectangle(
                    (obs["x_min"], obs["y_min"]), w, h, facecolor="black"
                )
                ax.add_patch(rpatch)

    def init_water_visual(self, ax):
        # Clear old references
        self.water_line_objects.clear()
        self._base_segments = []

        for y_line, segs in self.water_segments:
            for xs, xe in segs:
                (line_obj,) = ax.plot(
                    [xs, xe], [y_line, y_line], color="white", linewidth=2, zorder=1
                )
                self.water_line_objects.append(line_obj)
                # Store the current absolute positions so we can do incremental shifts.
                self._base_segments.append((y_line, xs, xe))

        # Reset the shift each time we init
        self._cumulative_shift = 0.0

        return self.water_line_objects

    def update_water_visual_incr(self, dt=0.02):
        """
        Incrementally shift each dash by water_drift_speed * dt.
        As soon as its right edge crosses x_max, we re-randomize
        the dash near x_min (no gap).
        Similarly, if its left edge crosses x_min, we re-randomize
        it near x_max, etc.
        """
        d_shift = self.water_drift_speed * dt
        width = self.x_max - self.x_min

        for i, line_obj in enumerate(self.water_line_objects):
            y_line, xs_abs, xe_abs = self._base_segments[i]

            # 1) shift them incrementally
            xs_new = xs_abs + d_shift
            xe_new = xe_abs + d_shift

            # 2) if the dash's right edge is beyond x_max => re-randomize at x_min
            #    That means we spawn a "fresh" dash so there's no gap in time
            if xe_new >= self.x_max:
                seg_len = np.random.uniform(0.1, 0.3)
                xs_new = self.x_min
                xe_new = xs_new + seg_len

            #  optionally handle the left boundary similarly:
            elif xs_new <= self.x_min:
                seg_len = np.random.uniform(0.1, 0.3)
                xe_new = self.x_max
                xs_new = xe_new - seg_len

            # 3) clamp partial overlaps to avoid "stretched" lines
            xs_cl = np.clip(xs_new, self.x_min, self.x_max)
            xe_cl = np.clip(xe_new, self.x_min, self.x_max)

            if xs_cl >= xe_cl:
                # Hide the dash if no valid overlap
                line_obj.set_data([], [])
            else:
                line_obj.set_data([xs_cl, xe_cl], [y_line, y_line])

            # 4) store them as the new absolute positions
            self._base_segments[i] = (y_line, xs_new, xe_new)

        return self.water_line_objects


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

    # Create tank object for visualization
    tank = Tank(drone_radius=DRONE_RADIUS, water_drift_speed=0.5)

    # Set limits for the tank - use tank dimensions
    tank_left = tank.x_min
    tank_right = tank.x_max
    tank_bottom = tank.y_min
    tank_top = tank.y_max

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

    # Draw tank static elements (walls, obstacles, water background)
    tank.draw_static(ax_traj)

    # Initialize water lines
    water_lines = tank.init_water_visual(ax_traj)

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

        # Initialize water lines
        for line in water_lines:
            line.set_data([], [])

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
                *water_lines,
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
                *water_lines,
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

        # Update water lines
        updated_water_lines = tank.update_water_visual_incr(dt=time_step_size)

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
                *updated_water_lines,
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
                *updated_water_lines,
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
