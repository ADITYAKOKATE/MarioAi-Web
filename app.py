import random
import time
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allow local testing HTML to hit Python API

class GridWorld:
    def __init__(self, size=15):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.mario = self.start
        self.level_num = 1
        self.walls = []
        self.enemies = []
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.generate_map()

    def generate_map(self):
        self.walls = []
        self.enemies = []
        self.mario = self.start
        
        # Procedurally generate 20 random walls
        for _ in range(20):
            x, y = random.randint(1, self.size-1), random.randint(1, self.size-1)
            # Leave the start and goal adjacent tiles slightly open
            if (x,y) not in self.walls and (x,y) != self.start and (x,y) != self.goal and (x,y) != (1,0) and (x,y) != (0,1) and (x,y) != (13,14) and (x,y) != (14,13):
                self.walls.append((x,y))
                
        # Procedurally generate 8 random Goomba threats
        for _ in range(8):
            x, y = random.randint(1, self.size-2), random.randint(1, self.size-2)
            if (x,y) not in self.walls and (x,y) not in self.enemies and (x,y) != self.start and (x,y) != self.goal:
                self.enemies.append((x,y))

    def reset_mario(self):
        self.mario = self.start
        return self.mario

    def step(self, action):
        x, y = self.mario
        dist_before = abs(self.goal[0] - x) + abs(self.goal[1] - y)
        
        if action == 'UP': y -= 1
        elif action == 'DOWN': y += 1
        elif action == 'LEFT': x -= 1
        elif action == 'RIGHT': x += 1

        if x < 0 or x >= self.size or y < 0 or y >= self.size or (x,y) in self.walls:
            return self.mario, -10, False # Wall penalty

        self.mario = (x, y)
        dist_after = abs(self.goal[0] - x) + abs(self.goal[1] - y)

        if self.mario == self.goal:
            return self.mario, 500, True # Reached Flag
            
        if self.mario in self.enemies:
            return self.mario, -200, True # Hit Goomba

        # Act accordingly based on the coordinates of the goal
        if dist_after > dist_before:
            reward = -15 # Strong penalty for moving away from goal coordinates
        else:
            reward = 10  # Reward for moving closer
            
        return self.mario, reward, False

class QAgent:
    def __init__(self, env):
        self.env = env
        self.start_time = time.time()
        self.reset_learning()

    def reset_learning(self):
        self.q_table = {}
        self.alpha = 0.5
        self.gamma = 0.95
        self.epsilon = 0.05 
        self.episode = 1
        self.total_reward = 0
        self.steps_taken = 0
        self.reward_history = [] 

    def get_q(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {}
            for a in self.env.actions:
                x, y = state
                if a == 'UP': y -= 1
                elif a == 'DOWN': y += 1
                elif a == 'LEFT': x -= 1
                elif a == 'RIGHT': x += 1
                
                # Heuristic Initialization: The AI mathematically "sees" the goal
                dist = abs(self.env.goal[0] - x) + abs(self.env.goal[1] - y)
                self.q_table[state][a] = -dist * 5.0
                
        return self.q_table[state][action]

    def choose_action(self, state):
        gx, gy = self.env.goal
        x, y = state
        
        # Goal Coordinate Awareness: If adjacent to the goal, act accordingly and step into it directly
        if abs(gx - x) + abs(gy - y) == 1:
            if gx > x: return 'RIGHT'
            if gx < x: return 'LEFT'
            if gy > y: return 'DOWN'
            if gy < y: return 'UP'

        if random.random() < self.epsilon:
            # Smart exploration: Bias random actions towards the goal coordinates
            if random.random() < 0.6: 
                opts = []
                if gx > x: opts.append('RIGHT')
                elif gx < x: opts.append('LEFT')
                if gy > y: opts.append('DOWN')
                elif gy < y: opts.append('UP')
                if opts: return random.choice(opts)
            return random.choice(self.env.actions)
            
        self.get_q(state, self.env.actions[0]) # ensure state init
        
        max_q = max(self.q_table[state].values())
        best_actions = [a for a, q in self.q_table[state].items() if q == max_q]
        return random.choice(best_actions)

    def learn_step(self):
        state = self.env.mario
        action = self.choose_action(state)
        next_state, reward, done = self.env.step(action)
        
        current_q = self.get_q(state, action)
        
        if done and reward > 0:
            max_next_q = 0 # Terminal success
        elif done and reward < 0:
            max_next_q = 0 # Terminal fail
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0.0 for a in self.env.actions}
            max_next_q = max(self.q_table[next_state].values())
            
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self.total_reward += reward
        self.steps_taken += 1
        
        res = {
            "x": next_state[0],
            "y": next_state[1],
            "reward": reward,
            "done": done,
            "episode": self.episode,
            "total_reward": round(self.total_reward, 2),
            "steps": self.steps_taken,
            "level": self.env.level_num,
            "history": self.reward_history,
            "time_alive": round(time.time() - self.start_time, 1),
            "goal_reached": (reward == 500)
        }
        
        if done or self.steps_taken > 200: 
            self.reward_history.append(round(self.total_reward, 2))
            if len(self.reward_history) > 50:
                self.reward_history.pop(0) # Keep last 50 episodes for graph
                
            if reward == 500: # Reached goal!
                self.env.level_num += 1
                self.env.generate_map()
                self.reset_learning() # Wipe brain for the new puzzle
            else:
                self.episode += 1
                self.total_reward = 0
                self.steps_taken = 0
                self.env.reset_mario()
                if self.episode % 3 == 0 and self.epsilon > 0.05:
                    self.epsilon *= 0.8
                
        return res

env = GridWorld(15)
agent = QAgent(env)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/state', methods=['GET'])
def get_state():
    return jsonify({
        "size": env.size,
        "mario": env.mario,
        "goal": env.goal,
        "enemies": env.enemies,
        "walls": env.walls,
        "level": env.level_num,
        "episode": agent.episode,
        "epsilon": round(agent.epsilon, 3),
        "history": agent.reward_history,
        "time_alive": round(time.time() - agent.start_time, 1)
    })

@app.route('/step', methods=['POST'])
def step():
    result = agent.learn_step()
    return jsonify(result)

@app.route('/reset', methods=['POST'])
def reset():
    global agent, env
    env = GridWorld(15)
    agent = QAgent(env)
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
