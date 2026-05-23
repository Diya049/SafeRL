import numpy as np
import random
from config import *
import cv2

class GridWorld:
    def __init__(self):
        self.size = GRID_SIZE
        self.reset_static()

    def reset_static(self):
        self.goal_pos = (self.size - 1, self.size - 1)
        self.obstacles = set()
        for i in range(2, 10):
            self.obstacles.add((i, 5))
        for j in range(3, 10):
            self.obstacles.add((7, j))
        self.obstacles.discard((4, 5))
        self.obstacles.discard((7, 6))
        self.hazards = {(8,1),(8,2),(9,1),(9,2),(2,8),(3,8),(2,9),(3,9)}

    def reset(self):
        self.agent_pos = [0, 0]
        self.humans = [(random.randint(0,10), random.randint(0,10)) for _ in range(3)]
        return self.get_state()

    def get_state(self):
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        goal_dx = np.sign(gx - ax)
        goal_dy = np.sign(gy - ay)
        hx, hy = min(self.humans, key=lambda h: abs(h[0]-ax)+abs(h[1]-ay))
        return (goal_dx, goal_dy, np.sign(hx-ax), np.sign(hy-ay))

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.size and 0 <= y < self.size and pos not in self.obstacles

    def move_humans(self):
        new = []
        for (x, y) in self.humans:
            dx, dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
            nx, ny = x+dx, y+dy
            new.append((nx,ny) if self.is_valid((nx,ny)) else (x,y))
        self.humans = new

    def step(self, action):
        dx, dy = [(-1,0),(1,0),(0,-1),(0,1),(0,0)][action]
        nx = self.agent_pos[0] + dx
        ny = self.agent_pos[1] + dy
        if self.is_valid((nx, ny)):
            self.agent_pos = [nx, ny]
        self.move_humans()
        reward = STEP_COST
        done = False
        info = {"hazard": False, "collision": False, "distance": False}
        dist = abs(nx - self.goal_pos[0]) + abs(ny - self.goal_pos[1])
        reward += -0.2 * dist
        if tuple(self.agent_pos) == self.goal_pos:
            reward += GOAL_REWARD
            done = True
        if tuple(self.agent_pos) in self.hazards:
            reward -= 40
            done = True
            info["hazard"] = True
        if tuple(self.agent_pos) in self.humans:
            reward -= 50
            done = True
            info["collision"] = True
        for h in self.humans:
            if abs(h[0]-nx) + abs(h[1]-ny) < 2:
                reward -= 5
                info["distance"] = True
        return self.get_state(), reward, done, info

    def render(self):
        grid = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        grid[:, :] = [255, 218, 185]
        for x, y in self.obstacles:
            grid[x, y] = [120, 120, 120]
        for x, y in self.hazards:
            grid[x, y] = [176, 224, 230]
        if hasattr(self, "humans"):
            for x, y in self.humans:
                grid[x, y] = [255, 140, 0]
        gx, gy = self.goal_pos
        grid[gx, gy] = [0, 255, 0]
        ax, ay = self.agent_pos
        grid[ax, ay] = [50, 205, 50]
        return grid
