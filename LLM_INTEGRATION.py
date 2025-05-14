import pygame
import sys
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import heapq
import re

# Initialize LLM (Free GPT-2 for simulation) with offline mode
try:
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2", local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained("distilgpt2", local_files_only=True)
except OSError:
    print("Model files not found locally. Please download 'distilgpt2' beforehand.")
    sys.exit(1)
model.eval()
pad_token_id = tokenizer.eos_token_id

# A* Pathfinding Algorithm
def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
def neighbors(node, obstacles):
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
    result = []
    for dx,dy in dirs:
        nx, ny = node[0]+dx, node[1]+dy
        if 0<=nx<COLS and 0<=ny<ROWS and (nx,ny) not in obstacles:
            result.append((nx, ny))
    return result

def a_star(start, goal, obstacles):
    open_set=[]; heapq.heappush(open_set,(0, start)); came_from={}; g_score={start:0}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current==goal:
            path=[]; node=current
            while node in came_from:
                path.append(node); node=came_from[node]
            return path[::-1]
        for nb in neighbors(current, obstacles):
            tent=g_score[current]+1
            if nb not in g_score or tent<g_score[nb]:
                came_from[nb]=current; g_score[nb]=tent
                f=tent+heuristic(nb,goal)
                heapq.heappush(open_set,(f,nb))
    return []

# Sync LLM parse: returns (x,y) or None
TARGET_REGEX=re.compile(r"\((\d+),(\d+)\)")
def query_target(user_text):
    prompt=(
        f"You are controlling a robot on a grid of size {COLS}x{ROWS}. "
        f"The box is at {tuple(box)}.\n"
        "Interpret the user's instruction (e.g., 'move to the box', 'go to (5,3)'). "
        "Respond with coordinates in format '(x,y)'."
        f"\nUser: '{user_text}'\nResponse:"
    )
    inputs=tokenizer.encode(prompt,return_tensors="pt")
    attention_mask=torch.ones_like(inputs)
    outputs=model.generate(
        inputs, attention_mask=attention_mask, max_new_tokens=10,
        num_beams=3, pad_token_id=pad_token_id
    )
    decoded=tokenizer.decode(outputs[0],skip_special_tokens=True)
    print("LLM raw response:", decoded)
    m=TARGET_REGEX.search(decoded)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None

# Pygame init
pygame.init()
WIDTH, HEIGHT, GRID=800,600,40
ROWS, COLS=HEIGHT//GRID, WIDTH//GRID
WHITE,BLACK,RED,GRAY=(255,255,255),(0,0,0),(255,0,0),(169,169,169)
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Robot with LLM Control")
clock=pygame.time.Clock()
font=pygame.font.SysFont(None,24)

# World state
original_box_pos=(7,7)
start_pos=(2,2)
robot=list(start_pos)
box=list(original_box_pos)
obstacles=[]

# Control state
phase='auto_pickup'
manual_text=''
full_path=[]
step_idx=0
prompt_text='Enter instruction:'

running=True
while running:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            running=False
        elif phase=='manual' and event.type==pygame.KEYDOWN:
            if event.key==pygame.K_RETURN:
                text=manual_text.strip()
                manual_text=''
                if not text:
                    continue
                # direct box command fallback
                if 'box' in text.lower():
                    target=tuple(box)
                else:
                    target=query_target(text)
                print("Parsed target:", target)
                if target:
                    full_path=a_star(tuple(robot), target, obstacles)
                    step_idx=0
            elif event.key==pygame.K_BACKSPACE:
                manual_text=manual_text[:-1]
            else:
                manual_text+=event.unicode

    # Auto pickup
    if phase=='auto_pickup':
        path=a_star(tuple(robot), tuple(box), obstacles)
        for s in path: robot=list(s)
        phase='auto_return'
    # Auto return
    elif phase=='auto_return':
        path=a_star(tuple(robot), start_pos, obstacles)
        for s in path: robot=list(s)
        phase='manual'
    # Manual follow
    elif phase=='manual' and step_idx < len(full_path):
        robot=list(full_path[step_idx]); step_idx+=1

    # Draw
    screen.fill(WHITE)
    for x in range(COLS):
        for y in range(ROWS): pygame.draw.rect(screen,GRAY,(x*GRID,y*GRID,GRID,GRID),1)
    pygame.draw.rect(screen,RED,(box[0]*GRID,box[1]*GRID,GRID,GRID))
    pygame.draw.rect(screen,BLACK,(robot[0]*GRID,robot[1]*GRID,GRID,GRID))
    if phase=='manual':
        screen.blit(font.render(prompt_text,True,BLACK),(10,HEIGHT-60))
        pygame.draw.rect(screen,BLACK,(10,HEIGHT-40,780,30),1)
        screen.blit(font.render(manual_text,True,BLACK),(15,HEIGHT-38))

    pygame.display.flip(); clock.tick(5)

pygame.quit(); sys.exit()
