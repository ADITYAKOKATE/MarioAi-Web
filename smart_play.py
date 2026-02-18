import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import MarioEnv

def make_env():
    def _init():
        env = MarioEnv(render_mode="human")
        return env
    return _init

def play(model_path="mario_ppo_final"):
    print(f"Loading model from {model_path}...")
    
    # Recreate environment (Same structure as training)
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    # Load Model (Try final, then interrupted, then checkpoints)
    model = None
    paths_to_try = [
        "mario_ppo_final",
        "mario_ppo_interrupted",
        "train/mario_model_200000_steps"
    ]
    
    for path in paths_to_try:
        try:
            print(f"Trying to load {path}...")
            model = PPO.load(path)
            print(f"Success! Loaded {path}")
            break
        except FileNotFoundError:
            continue
            
    if model is None:
        print("Error: No model found. Please train first.")
        return

    # Play Loop
    obs = env.reset()
    done = False
    total_reward = 0
    
    # Debug Helper Variables
    last_x_pos = 0
    stuck_counter = 0
    
    print("\n--- Starting Smart Evolution Loop ---")
    print("Columns: [Step] | Action | Position | Stuck Count | Heuristic Active?")
    
    try:
        step = 0
        while True:
            # 1. Get Model Prediction
            action, _states = model.predict(obs, deterministic=True)
            
            # 2. Heuristic Check (The "Smart" Part)
            # We access the internal env to get the player position for debugging/heuristics
            # env.envs[0] is the MarioEnv instance inside DummyVecEnv
            current_x = env.envs[0].player_pos[0]
            current_y = env.envs[0].player_pos[1]
            
            # Check if we moved horizontally
            if abs(current_x - last_x_pos) < 1:
                stuck_counter += 1
            else:
                stuck_counter = 0
                
            last_x_pos = current_x
            
            # Override Action if Stuck
            heuristic_active = False
            if stuck_counter > 10: # If stuck for 10 frames (approx 0.3s)
                action = [2] # FORCE JUMP
                heuristic_active = True
                # Reset counter slightly so we don't spam jump every single frame if it takes a moment to lift off
                if stuck_counter > 15: stuck_counter = 0 

            # --- HEURISTIC: PIT SAFETY ---
            # Gaps are at x=400, x=800, x=1300
            # We treat the model like a "copilot" but take control for safety
            
            # If we are RAPIDLY approaching a pit (within 20 pixels), JUMP!
            # Tightened triggers so we jump CLOSER to the edge
            if (380 < current_x < 400) or (780 < current_x < 800) or (1280 < current_x < 1300):
                 action = [2] # FORCE JUMP
                 heuristic_active = True
                 stuck_counter = 0

            # 3. Take Step
            obs, reward, done, info = env.step(action)
            env.render()
            
            # 4. Logs
            explanation = ""
            if heuristic_active:
                if action == [2]: explanation = "*** FORCE JUMP (Safety/Stuck) ***"
                
            act_str = ["Stay", "Right", "Jump"][int(action[0])]
            print(f"[{step:04d}] | {act_str:5} | Pos: ({current_x:.1f}, {current_y:.1f}) | {explanation}")
            
            # Sync variables
            total_reward += reward
            step += 1
            time.sleep(0.02) # Watchable speed
            
            if done:
                 print(f"Episode finished. Total Reward: {total_reward}")
                 
                 # Check if goal was reached (Reward > 90)
                 if total_reward > 90:
                    print("\n🏆 Goal Reached! Stopping script.")
                    break
                 
                 print("Goal not reached. Restarting level...")
                 obs = env.reset()
                 total_reward = 0
                 stuck_counter = 0
                 step = 0
                 print("-" * 50)
                 
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    env.close()

if __name__ == "__main__":
    play("mario_ppo_final")
