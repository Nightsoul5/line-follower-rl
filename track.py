import math


class Track:
    """Closed-loop track: wavy circle centerline with a fixed line width."""

    def __init__(self, center=(400, 300), radius=200, wave_amp=50,
                 wave_freq=3, width=20, num_points=200):
        self.width = width
        cx, cy = center
        self.points = []
        for i in range(num_points):
            theta = 2 * math.pi * i / num_points
            r = radius + wave_amp * math.sin(wave_freq * theta)
            self.points.append((cx + r * math.cos(theta),
                                cy + r * math.sin(theta)))

    def distance_to(self, x, y):
        """Minimum distance from point (x, y) to the centerline."""
        best = float("inf")
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]  # wrap → closed loop
            best = min(best, _point_segment_dist(x, y, x1, y1, x2, y2))
        return best

    def is_on_line(self, x, y):
        return self.distance_to(x, y) <= self.width / 2

    def start_pose(self):
        """Robot start: on the first centerline point, facing along the track."""
        return self.pose_at(0)

    def pose_at(self, index):
        """Pose on the centerline at the given point index, facing forward."""
        n = len(self.points)
        x0, y0 = self.points[index % n]
        x1, y1 = self.points[(index + 1) % n]
        angle = math.atan2(y1 - y0, x1 - x0)
        return x0, y0, angle

    def start_line(self, overhang=5):
        """Endpoints of the start/finish line: perpendicular stripe across the
        track at the start position, extending `overhang` px past each edge."""
        x0, y0, angle = self.start_pose()
        half = self.width / 2 + overhang
        px, py = math.cos(angle + math.pi / 2), math.sin(angle + math.pi / 2)
        return ((x0 - half * px, y0 - half * py),
                (x0 + half * px, y0 + half * py))


class Robot:
    """Kinematic robot: constant speed, three steering actions."""

    SPEED = 4.0          # px per step
    STEER_DELTA = 0.08   # rad per step
    LOOKAHEAD = 25.0     # sensor bar distance in front of robot (px)
    SENSOR_SPACING = 8.0 # lateral gap between sensors (px)
    NUM_SENSORS = 5

    def __init__(self, x=0.0, y=0.0, angle=0.0):
        self.reset(x, y, angle)

    def reset(self, x, y, angle):
        self.x, self.y, self.angle = x, y, angle

    def step(self, steer):
        """steer: -1 = left, 0 = straight, +1 = right."""
        self.angle += steer * self.STEER_DELTA
        self.x += self.SPEED * math.cos(self.angle)
        self.y += self.SPEED * math.sin(self.angle)

    def sensor_positions(self):
        """5 points on a bar in front of the robot, perpendicular to heading."""
        cx = self.x + self.LOOKAHEAD * math.cos(self.angle)
        cy = self.y + self.LOOKAHEAD * math.sin(self.angle)
        px = math.cos(self.angle + math.pi / 2)   # perpendicular unit vector
        py = math.sin(self.angle + math.pi / 2)
        half = (self.NUM_SENSORS - 1) / 2         # offsets -2..+2
        return [(cx + (i - half) * self.SENSOR_SPACING * px,
                 cy + (i - half) * self.SENSOR_SPACING * py)
                for i in range(self.NUM_SENSORS)]

    def read_sensors(self, track):
        """Binary readings: [s0..s4], 1 if that sensor point is on the line."""
        return [1 if track.is_on_line(sx, sy) else 0
                for sx, sy in self.sensor_positions()]


def _point_segment_dist(px, py, x1, y1, x2, y2):
    """Distance from point P to segment (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    # projection parameter of P onto the segment, clamped to [0, 1]
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
