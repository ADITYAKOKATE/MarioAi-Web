import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import cv2

class MarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window_width = 256
        self.window_height = 240
        self.level_width = 2000 # Much longer level
        self.screen_dim = (self.window_width, self.window_height)
        
        # Action space: 0: Stay/Left, 1: Right, 2: Jump
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 84x84 grayscale image
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
        # Pygame setup
        pygame.init()
        pygame.display.init()
        self.window = None
        self.clock = None
        self.surf = pygame.Surface((self.level_width, self.window_height)) # Draw on full level surface
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_pos = [20, 180]
        self.player_vel = [0, 0]
        self.on_ground = False
        self.steps = 0
        self.terminated = False
        self.stuck_counter = 0
        self.camera_x = 0
        
        # Level Design (Longer & Harder)
        self.goal_pos = [1900, 180] # Far right
        self.obstacles = []
        
        # 1. Floor (with gaps)
        # Ground segments: (start_x, width)
        ground_segments = [
            (0, 400),    # Start area
            (500, 300),  # Gap at 400-500
            (900, 400),  # Gap at 800-900
            (1450, 600)  # Gap at 1300-1450
        ]
        
        for x, w in ground_segments:
            self.obstacles.append(pygame.Rect(x, 220, w, 20))
            
        # 2. Platforms & Blocks
        platforms = [
            # First obstacle set
            (200, 190, 30, 30),   # Low block
            (280, 150, 30, 30),   # High block
            
            # Crossing the first gap
            (420, 160, 60, 10),   # Bridge platform
            
            # Second section
            (600, 190, 30, 30),
            (630, 190, 30, 30),   # Double block
            (750, 140, 60, 10),   # High platform
            
            # Crossing second gap
            (850, 170, 40, 10),   # Stepping stone
            
            # Third section (Stairs)
            (1000, 190, 30, 30),
            (1030, 160, 30, 30),
            (1060, 130, 30, 30),
            
            # Final stretch
            (1350, 160, 80, 10),  # Long jump platform
            (1600, 190, 30, 30),  # Final hurdle
        ]
        
        for x, y, w, h in platforms:
            self.obstacles.append(pygame.Rect(x, y, w, h))
            
        # Walls
        self.obstacles.append(pygame.Rect(-10, 0, 10, 240)) # Left wall
        self.obstacles.append(pygame.Rect(2000, 0, 10, 240)) # Right wall
        
        # Decoration Clouds
        self.clouds = [(x, np.random.randint(20, 100)) for x in range(50, 2000, 150)]
        
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        
        # Current Physics Config
        SPEED = 5 
        JUMP_FORCE = -15 # Even higher jump
        GRAVITY = 0.8
        
        # Frame Skip Loop
        for _ in range(4):
            if terminated: break
            self.steps += 1
            reward = 0
            prev_x = self.player_pos[0]
            
            # Movement
            if action == 1: self.player_vel[0] = SPEED
            elif action == 0: self.player_vel[0] = 0
            
            if action == 2 and self.on_ground:
                self.player_vel[1] = JUMP_FORCE
                self.on_ground = False
                
            self.player_vel[1] += GRAVITY
            if self.player_vel[1] > 10: self.player_vel[1] = 10

            # X Collision
            self.player_pos[0] += self.player_vel[0]
            player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], 14, 14)
            for obs in self.obstacles:
                if player_rect.colliderect(obs):
                    if self.player_vel[0] > 0: self.player_pos[0] = obs.left - 14
                    elif self.player_vel[0] < 0: self.player_pos[0] = obs.right
                    player_rect.x = self.player_pos[0]

            # Y Collision
            self.on_ground = False
            self.player_pos[1] += self.player_vel[1]
            player_rect.y = self.player_pos[1]
            for obs in self.obstacles:
                if player_rect.colliderect(obs):
                    if self.player_vel[1] > 0:
                        self.player_pos[1] = obs.top - 14
                        self.player_vel[1] = 0
                        self.on_ground = True
                    elif self.player_vel[1] < 0:
                        self.player_pos[1] = obs.bottom
                        self.player_vel[1] = 0
                    player_rect.y = self.player_pos[1]

            # Camera Follow
            self.camera_x = self.player_pos[0] - 100
            if self.camera_x < 0: self.camera_x = 0
            if self.camera_x > self.level_width - self.window_width: self.camera_x = self.level_width - self.window_width

            # Rewards
            dist_moved = self.player_pos[0] - prev_x
            if dist_moved > 0:
                reward += dist_moved * 0.1
                self.stuck_counter = 0
            else:
                self.stuck_counter += 1
                
            if self.stuck_counter > 20: reward -= 0.1
            
            # Goal
            goal_rect = pygame.Rect(self.goal_pos[0], self.goal_pos[1], 20, 40)
            if player_rect.colliderect(goal_rect):
                reward += 100
                terminated = True
                
            # Death (Pit)
            if self.player_pos[1] > self.window_height:
                reward -= 50
                terminated = True # Die

            if self.steps > 4000: truncated = True # Longer timeout for longer level
            
            total_reward += reward
            self.terminated = terminated

        if self.render_mode == "human":
            self.render()

        observation = self._get_obs()
        return observation, total_reward, terminated, truncated, {}

    def _get_obs(self):
        # Draw everything to the full level surface
        self.surf.fill((107, 140, 255)) 
        
        # Scenery
        for cx, cy in self.clouds:
             pygame.draw.circle(self.surf, (255, 255, 255), (cx, cy), 20)
             pygame.draw.circle(self.surf, (255, 255, 255), (cx+15, cy-5), 25)
        
        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(self.surf, (200, 76, 12), obs)
            pygame.draw.rect(self.surf, (0,0,0), obs, 2) 

        # Goal
        pole = pygame.Rect(self.goal_pos[0]+8, self.goal_pos[1], 4, 40)
        flag = pygame.Rect(self.goal_pos[0]+12, self.goal_pos[1], 15, 10)
        pygame.draw.rect(self.surf, (255, 255, 255), pole)
        pygame.draw.rect(self.surf, (50, 205, 50), flag)

        # Mario
        x, y = self.player_pos
        pygame.draw.rect(self.surf, (255, 0, 0), (x, y, 14, 14))
        pygame.draw.rect(self.surf, (255, 200, 150), (x+2, y+2, 10, 8)) # Face
        
        # Create Viewport (Camera)
        # We act like a camera by creating a sub-surface
        view_rect = pygame.Rect(self.camera_x, 0, self.window_width, self.window_height)
        viewport = self.surf.subsurface(view_rect)
        
        # Convert viewport to observation
        view_str = pygame.image.tostring(viewport, "RGB")
        img = np.frombuffer(view_str, dtype=np.uint8)
        img = img.reshape((self.window_height, self.window_width, 3))
        
        # For the AGENT, we center the view on the player so it generalizes better
        # Actually, for PPO reuse, we should just downscale the viewport like before
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        return img[:, :, None]

    def render(self):
        if self.window is None and self.render_mode == "human":
            self.window = pygame.display.set_mode(self.screen_dim)
            self.clock = pygame.time.Clock()
        
        if self.window is not None:
            # Blit the camera view to the window
            view_rect = pygame.Rect(self.camera_x, 0, self.window_width, self.window_height)
            self.window.blit(self.surf, (0, 0), area=view_rect)
            
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
