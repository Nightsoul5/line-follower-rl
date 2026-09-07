import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from track import Track, Robot

LAPS_TO_WIN = 2
PROGRESS_SCALE = 50.0   # reward per radian of forward progress (progress mode)


class LineFollowerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=2500, random_start=False,
                 reward_mode="progress", track_kwargs=None):
        super().__init__()
        assert reward_mode in ("progress", "presence")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.random_start = random_start
        self.reward_mode = reward_mode

        kwargs = {"center": (400, 300)}
        if track_kwargs:
            kwargs.update(track_kwargs)
        self.track = Track(**kwargs)
        self._track_center = kwargs["center"]
        self.robot = Robot()

        # --- RL interface ---
        self.observation_space = spaces.MultiBinary(Robot.NUM_SENSORS)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=straight, 2=right

        # lap tracking (angle traveled around the track center)
        self._track_center = (400, 300)
        self._prev_angle = None
        self._angle_travelled = 0.0
        self.laps = 0
        self.steps = 0

        self._screen = None  # lazy pygame init for render_mode="human"

    # ------------------------------------------------------------------ API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start:
            index = int(self.np_random.integers(0, len(self.track.points)))
            self.robot.reset(*self.track.pose_at(index))
        else:
            self.robot.reset(*self.track.start_pose())
        self._prev_angle = self._robot_polar_angle()
        self._angle_travelled = 0.0
        self.laps = 0
        self.steps = 0
        obs = self._observe()
        return obs, {}

    def step(self, action):
        steer = int(action) - 1        # 0,1,2 -> -1,0,+1
        self.robot.step(steer)
        self.steps += 1
        obs = self._observe()

        terminated = False
        truncated = False
        reward = 0.0
        info = {"laps": self.laps}

        # --- reward ---
        progress = self._forward_progress()   # signed angle advanced this step
        if obs.sum() == 0:
            # line completely lost -> fail
            reward = -100.0
            terminated = True
            info["result"] = "line_lost"
        elif self.reward_mode == "progress":
            reward = PROGRESS_SCALE * progress          # pay for moving forward
            if obs[Robot.NUM_SENSORS // 2] != 1:
                reward -= 0.1                           # not centered
        else:  # "presence" (v1)
            if obs[Robot.NUM_SENSORS // 2] == 1:
                reward = 1.0               # centered on the line
            else:
                reward = -0.1              # on the line, but drifting off-center

        # --- lap tracking & win condition ---
        if not terminated:
            self._angle_travelled += progress
            if self._angle_travelled >= 2 * math.pi:
                self._angle_travelled -= 2 * math.pi
                self.laps += 1
                reward += 100.0
                info["laps"] = self.laps
                if self.laps >= LAPS_TO_WIN:
                    terminated = True
                    info["result"] = "win"

        # --- truncation (safety cap) ---
        if not terminated and self.steps >= self.max_steps:
            truncated = True
            info["result"] = "max_steps"

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------- helpers

    def _observe(self):
        return np.array(self.robot.read_sensors(self.track), dtype=np.int8)

    def _robot_polar_angle(self):
        cx, cy = self._track_center
        return math.atan2(self.robot.y - cy, self.robot.x - cx)

    def _forward_progress(self):
        """Signed angle advanced around the track center this step (wrap-safe).
        Positive = forward along the track, negative = backward."""
        angle = self._robot_polar_angle()
        delta = angle - self._prev_angle
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        self._prev_angle = angle
        return delta

    # -------------------------------------------------------------- render

    def render(self):
        if self.render_mode != "human":
            return
        import pygame
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((800, 600))
            self._clock = pygame.time.Clock()
            surf = pygame.Surface((800, 600))
            surf.fill((25, 25, 30))
            pts = [(int(x), int(y)) for x, y in self.track.points]
            pygame.draw.polygon(surf, (70, 70, 80), pts, width=self.track.width)
            pygame.draw.polygon(surf, (110, 110, 125), pts, width=1)
            (ax, ay), (bx, by) = self.track.start_line()
            pygame.draw.line(surf, (235, 235, 240),
                             (int(ax), int(ay)), (int(bx), int(by)), 4)
            self._track_surface = surf

        pygame.event.pump()
        self._screen.blit(self._track_surface, (0, 0))
        a = self.robot.angle
        nose = (self.robot.x + 14 * math.cos(a), self.robot.y + 14 * math.sin(a))
        tl = (self.robot.x + 10 * math.cos(a + 2.5), self.robot.y + 10 * math.sin(a + 2.5))
        tr = (self.robot.x + 10 * math.cos(a - 2.5), self.robot.y + 10 * math.sin(a - 2.5))
        pygame.draw.polygon(self._screen, (80, 160, 255), [nose, tl, tr])
        for (sx, sy), on in zip(self.robot.sensor_positions(),
                                self.robot.read_sensors(self.track)):
            color = (60, 220, 100) if on else (230, 70, 70)
            pygame.draw.circle(self._screen, color, (int(sx), int(sy)), 4)
        pygame.display.set_caption(f"Line Follower — lap {self.laps}/{LAPS_TO_WIN}")
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
