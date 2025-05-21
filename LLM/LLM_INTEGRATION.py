import pygame
import sys
import queue
import sounddevice as sd
import vosk
import json
import numpy as np
import os
import heapq

# Simulation settings
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
ROWS, COLS = HEIGHT // GRID_SIZE, WIDTH // GRID_SIZE
ROBOT_RADIUS = 20
OBSTACLE_COLOR = (255, 0, 0)
ROBOT_COLOR = (0, 255, 0)

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Offline Voice-Controlled Robot")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, grid):
    frontier = [(0, start)]
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            next_node = (current[0] + dx, current[1] + dy)
            if 0 <= next_node[0] < COLS and 0 <= next_node[1] < ROWS:
                if next_node == goal or not grid[next_node[1]][next_node[0]]:
                    new_cost = cost_so_far[current] + 1
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        priority = new_cost + heuristic(goal, next_node)
                        heapq.heappush(frontier, (priority, next_node))
                        came_from[next_node] = current

    path = []
    cur = goal
    while cur != start:
        if cur not in came_from:
            return []
        path.append((cur[0] * GRID_SIZE + GRID_SIZE // 2, cur[1] * GRID_SIZE + GRID_SIZE // 2))
        cur = came_from[cur]
    path.reverse()
    return path

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 5
        self.angle = 0
        self.current_action = None
        self.current_target = None
        self.clean_points = []
        self.path = []

    def move_forward(self):
        dx = self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).x
        dy = self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).y
        if not self.check_collision(self.x + dx, self.y + dy):
            self.x += dx
            self.y += dy

    def move_backward(self):
        dx = self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).x
        dy = self.speed * pygame.math.Vector2(1, 0).rotate(-self.angle).y
        if not self.check_collision(self.x - dx, self.y - dy):
            self.x -= dx
            self.y -= dy

    def turn_left(self):
        self.angle += 10

    def turn_right(self):
        self.angle -= 10

    def plan_path(self, target_pos):
        start = (int(self.x) // GRID_SIZE, int(self.y) // GRID_SIZE)
        goal = (int(target_pos[0]) // GRID_SIZE, int(target_pos[1]) // GRID_SIZE)
        print(f"[DEBUG] Planning path from {start} to {goal}")
        self.path = astar(start, goal, obstacle_grid)

    def follow_path(self):
        if self.path:
            target = pygame.math.Vector2(self.path[0])
            pos = pygame.math.Vector2(self.x, self.y)
            if pos.distance_to(target) < 5:
                self.path.pop(0)
            else:
                direction = (target - pos).normalize() * self.speed
                new_x = self.x + direction.x
                new_y = self.y + direction.y
                if not self.check_collision(new_x, new_y):
                    self.x = new_x
                    self.y = new_y

    def draw(self, screen):
        pygame.draw.circle(screen, ROBOT_COLOR, (int(self.x), int(self.y)), ROBOT_RADIUS)

    def check_collision(self, new_x, new_y):
        return False  # Disable all collision

    # --- Dirt feature: clean dirt method ---
    def clean_dirt(self):
        global dirt_patches
        if dirt_patches:
            if not self.path:
                nearest = min(dirt_patches, key=lambda p: pygame.math.Vector2(self.x, self.y).distance_to(p))
                self.plan_path(nearest)
            else:
                self.follow_path()
                if pygame.math.Vector2(self.x, self.y).distance_to(self.path[0]) < 10:
                    cleaned = self.path[0]
                    dirt_patches = [d for d in dirt_patches if pygame.math.Vector2(d).distance_to((self.x, self.y)) > 10]
                    print(f"[DEBUG] Cleaned dirt at {cleaned}")
                    self.path = []

            if not dirt_patches:
                print("All dirt cleaned.")
                self.current_action = None

# Obstacles
obstacles = [
    {'rect': pygame.Rect(100, 100, 100, 200), 'label': 'fridge'},
    {'rect': pygame.Rect(250, 100, 150, 50), 'label': 'sink'},
    {'rect': pygame.Rect(450, 100, 150, 100), 'label': 'stove'},
    {'rect': pygame.Rect(100, 350, 300, 100), 'label': 'counter'},
    {'rect': pygame.Rect(450, 250, 200, 200), 'label': 'table'},
]

obstacle_centers = {obs['label']: (obs['rect'].centerx, obs['rect'].centery) for obs in obstacles}

# --- Dirt feature: add dirt patches list ---
dirt_patches = [
    (150, 150),
    (300, 180),
    (500, 300),
    (200, 400),
]

# Grid with no obstacles
obstacle_grid = [[False for _ in range(COLS)] for _ in range(ROWS)]

robot = Robot(700, 500)

q = queue.Queue()
model_path = "vosk-model-small-en-us-0.15"
if not os.path.exists(model_path):
    print("Please download the Vosk model and place it in the same directory.")
    print("URL: https://alphacephei.com/vosk/models")
    sys.exit(1)
model = vosk.Model(model_path)

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

def setup_clean_points(target):
    if target in obstacle_centers:
        cx, cy = obstacle_centers[target]
        d = 40
        robot.clean_points = [(cx - d, cy - d), (cx + d, cy - d), (cx + d, cy + d), (cx - d, cy + d)]

def process_command_local(command):
    command = command.lower()
    print(f"[DEBUG] Received command: {command}")

    if 'stop' in command:
        robot.current_action = None
        robot.current_target = None
        robot.clean_points = []
        robot.path = []
        print("Robot has been stopped.")
        return

    directions = {
        'up': lambda: robot.move_forward(),
        'down': lambda: robot.move_backward(),
        'left': lambda: robot.turn_left(),
        'right': lambda: robot.turn_right(),
    }
    for dir_word, action_func in directions.items():
        if f"move {dir_word}" in command or dir_word == command.strip():
            action_func()
            return

    actions = {
        'move': ['move', 'go', 'walk', 'run', 'head', 'proceed', 'travel'],
        'clean': ['clean', 'wipe', 'scrub', 'sanitize', 'polish']
    }

    action_found = None
    for action, keywords in actions.items():
        if any(kw in command for kw in keywords):
            action_found = action
            break

    target = None
    for name in obstacle_centers:
        if name in command:
            target = name
            break

    # --- Dirt cleaning command ---
    if "clean dirt" in command or "clean debris" in command:
        robot.current_action = "clean_dirt"
        return

    print(f"[DEBUG] Action: {action_found}, Target: {target}")

    if action_found and target is None:
        print(f"Action '{action_found}' requires a target.")
        return

    robot.current_action = action_found
    robot.current_target = target
    if action_found == 'move':
        robot.plan_path(obstacle_centers[target])
    elif action_found == 'clean':
        setup_clean_points(target)

input_box = pygame.Rect(10, HEIGHT - 40, 780, 30)
input_text = ''
input_active = True

try:
    device = sd.default.device[0]
    samplerate = int(sd.query_devices(device, 'input')['default_samplerate'])
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                           dtype='int16', channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)

        while True:
            screen.fill((255, 255, 255))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and input_active:
                    if event.key == pygame.K_RETURN:
                        process_command_local(input_text)
                        input_text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode

            if not q.empty():
                data = q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    command = result.get("text", "")
                    if command:
                        print("Voice command:", command)
                        process_command_local(command)

            # --- Draw obstacles ---
            for obs in obstacles:
                pygame.draw.rect(screen, OBSTACLE_COLOR, obs['rect'])
                label_surface = font.render(obs['label'], True, (0, 0, 0))
                screen.blit(label_surface, (obs['rect'].x, obs['rect'].y - 20))

            # --- Draw dirt patches ---
            for dirt in dirt_patches:
                pygame.draw.circle(screen, (139, 69, 19), (int(dirt[0]), int(dirt[1])), 5)

            # --- Robot actions ---
            if robot.current_action == 'move' and robot.path:
                robot.follow_path()
            elif robot.current_action == 'clean' and robot.clean_points:
                if not robot.path and robot.clean_points:
                    robot.plan_path(robot.clean_points[0])
                elif robot.path:
                    robot.follow_path()
                    if pygame.math.Vector2(robot.x, robot.y).distance_to(robot.clean_points[0]) < 10:
                        robot.clean_points.pop(0)
                        robot.path = []

                if not robot.clean_points:
                    print(f"Finished cleaning the {robot.current_target}.")
                    robot.current_action = None
                    robot.current_target = None
            # --- Dirt cleaning action ---
            elif robot.current_action == 'clean_dirt':
                robot.clean_dirt()

            robot.draw(screen)

            pygame.draw.rect(screen, (200, 200, 200), input_box)
            txt_surface = font.render(input_text, True, (0, 0, 0))
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
            pygame.draw.rect(screen, (0, 0, 0), input_box, 2)

            pygame.display.flip()
            clock.tick(30)

except KeyboardInterrupt:
    print("Voice control interrupted by user")
except Exception as e:
    print("Voice input error:", e)
finally:
    pygame.quit()