from constants import (
    TIME_STEP_SIZE,
    DRONE_MASS,
    DRONE_INERTIA,
    DRAG_COEFF,
    DRONE_RADIUS,
    OFFSET_LAT,
    MAX_F_LONG,
    MAX_F_LAT,
    GRAVITY,
    TOP_Y,
    HOLE_WIDTH,
)
import numpy as np
from matplotlib.patches import Polygon, Circle


class UnderwaterDrone:
    def __init__(
        self,
        x=0.0,
        y=0.0,
        theta=0.0,
        v_x=0.0,
        v_y=0.0,
        omega=0.0,
        m=DRONE_MASS,
        I=DRONE_INERTIA,
        Cd=DRAG_COEFF,
        radius=DRONE_RADIUS,
        offset_lateral=OFFSET_LAT,
        gravity=GRAVITY,
        max_F_long=MAX_F_LONG,
        max_F_lat=MAX_F_LAT,
        top_y=TOP_Y,
        hole_width=HOLE_WIDTH,
    ):
        self.x = x
        self.y = y
        self.theta = theta
        self.v_x = v_x
        self.v_y = v_y
        self.omega = omega

        self.m = m
        self.I = I
        self.Cd = Cd
        self.radius = radius
        self.offset_lateral = offset_lateral
        self.gravity = gravity

        # Control bounds
        self.max_F_long = max_F_long  # max forward/backward thrust
        self.max_F_lat = max_F_lat  # max lateral thrust

        # To detect getting into air
        self.top_y = top_y
        self.hole_width = hole_width

        # We'll keep a "frozen" flag
        self.frozen = False

        # Visualization patches will be created later in init_visual()
        self.nose_patch = None
        self.body_patch = None

        # Prepare local geometry for the "nose" polygon
        self.nose_len = 0.3
        self.nose_half_w = 0.2
        # local coords: tip at (nose_len, 0), base corners around (0, +/- nose_half_w)
        self.nose_local_coords = np.array(
            [[self.nose_len, 0.0], [0.0, self.nose_half_w], [0.0, -self.nose_half_w]]
        )

    def step(self, action, dt=TIME_STEP_SIZE):
        """
        action = (F_long, F_lat)
          F_long: thrust in the drone's longitudinal direction (body x-axis).
          F_lat:  thrust in the drone's lateral direction (body y-axis).
        dt: time step for Euler integration.
        """
        if self.frozen:
            return

        if self._in_hole():
            self._freeze()
            return

        F_long, F_lat = action
        F_long = np.clip(F_long, -self.max_F_long, self.max_F_long)
        F_lat = np.clip(F_lat, -self.max_F_lat, self.max_F_lat)

        c = np.cos(self.theta)
        s = np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])

        thrust_body = np.array([F_long, F_lat])
        thrust_inertial = R @ thrust_body

        v = np.array([self.v_x, self.v_y])
        speed = np.linalg.norm(v)
        if speed > 1e-6:
            drag_dir = -v / speed
        else:
            drag_dir = np.array([0.0, 0.0])
        F_drag = self.Cd * speed**2 * drag_dir

        F_net = thrust_inertial + F_drag + np.array([0.0, -self.gravity])

        tau = self.offset_lateral * F_lat

        a_x = F_net[0] / self.m
        a_y = F_net[1] / self.m
        alpha = tau / self.I

        self.v_x += a_x * dt
        self.v_y += a_y * dt
        self.omega += alpha * dt

        self.x += self.v_x * dt
        self.y += self.v_y * dt
        self.theta += self.omega * dt

        # After integration, check if we *just* crossed into the hole region
        if self._in_hole():
            self._freeze()

    def _in_hole(self):
        """
        Return True if the drone's center is at/above top_y
        AND within the horizontal hole region, i.e. x in [-hole_half, hole_half].
        """
        hole_half = self.hole_width / 2.0
        if self.y >= self.top_y:
            if -hole_half <= self.x <= hole_half:
                return True
        return False

    def _freeze(self):
        """Freeze the drone's motion. Zero velocities, 'frozen'=True."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.omega = 0.0
        self.frozen = True

    def state(self):
        """Return (x, y, theta, v_x, v_y, omega)."""
        return (self.x, self.y, self.theta, self.v_x, self.v_y, self.omega)

    # Visualization methods
    def init_visual(self, ax):
        """Create the patch objects and add them to 'ax'."""
        # Nose patch (triangle)
        self.nose_patch = Polygon(
            self.nose_local_coords, closed=True, facecolor="darkred"
        )
        ax.add_patch(self.nose_patch)

        # Body patch (circle)
        self.body_patch = Circle((0, 0), radius=self.radius, facecolor="darkgreen")
        ax.add_patch(self.body_patch)

    def update_visual(self):
        """Update the patch positions according to the drone's current state."""
        x, y, theta, _, _, _ = self.state()

        # Rotate/translate the nose polygon
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        tri_world = (R @ self.nose_local_coords.T).T + np.array([x, y])
        self.nose_patch.set_xy(tri_world)

        # Update body
        self.body_patch.center = (x, y)
