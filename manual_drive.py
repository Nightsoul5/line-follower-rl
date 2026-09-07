import math
import pygame
from track import Track, Robot

WIDTH, HEIGHT = 800, 600
BG = (25, 25, 30)
LINE_COLOR = (70, 70, 80)
CENTER_COLOR = (110, 110, 125)
ROBOT_COLOR = (80, 160, 255)
ON_COLOR = (60, 220, 100)
OFF_COLOR = (230, 70, 70)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    track = Track(center=(WIDTH // 2, HEIGHT // 2))
    robot = Robot()
    robot.reset(*track.start_pose())

    # Pre-render the track once (static background)
    track_surface = pygame.Surface((WIDTH, HEIGHT))
    track_surface.fill(BG)
    pts = [(int(x), int(y)) for x, y in track.points]
    pygame.draw.polygon(track_surface, LINE_COLOR, pts, width=track.width)
    pygame.draw.polygon(track_surface, CENTER_COLOR, pts, width=1)
    (ax, ay), (bx, by) = track.start_line()
    pygame.draw.line(track_surface, (235, 235, 240),
                     (int(ax), int(ay)), (int(bx), int(by)), 4)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    robot.reset(*track.start_pose())

        keys = pygame.key.get_pressed()
        steer = 0
        if keys[pygame.K_LEFT]:
            steer = -1
        elif keys[pygame.K_RIGHT]:
            steer = 1
        robot.step(steer)

        sensors = robot.read_sensors(track)

        # --- draw ---
        screen.blit(track_surface, (0, 0))

        # Robot body: triangle pointing along heading
        a = robot.angle
        nose = (robot.x + 14 * math.cos(a), robot.y + 14 * math.sin(a))
        tail_l = (robot.x + 10 * math.cos(a + 2.5), robot.y + 10 * math.sin(a + 2.5))
        tail_r = (robot.x + 10 * math.cos(a - 2.5), robot.y + 10 * math.sin(a - 2.5))
        pygame.draw.polygon(screen, ROBOT_COLOR, [nose, tail_l, tail_r])

        # Sensors
        for (sx, sy), on in zip(robot.sensor_positions(), sensors):
            pygame.draw.circle(screen, ON_COLOR if on else OFF_COLOR,
                               (int(sx), int(sy)), 4)

        pygame.display.set_caption(f"Line Follower — sensors: {sensors}")
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
