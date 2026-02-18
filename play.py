import time
import gymnasium as gym
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
    
    # Recreate the environment with render_mode='human'
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    # Load the trained model
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        # Try loading the interrupted model if the final one doesn't exist
        fallback_path = "mario_ppo_interrupted"
        print(f"Model '{model_path}' not found. Trying '{fallback_path}'...")
        try:
            model = PPO.load(fallback_path)
        except FileNotFoundError:
             print(f"Error: Neither '{model_path}.zip' nor '{fallback_path}.zip' found.")
             print("Please run 'python train.py' and let it run for at least a few seconds!")
             return

    # Play the game
    obs = env.reset()
    done = False
    total_reward = 0
    
    print("Starting evaluation loop... Press Ctrl+C to stop.")
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            env.render()
            total_reward += reward
            time.sleep(0.01) # Slow down for human viewing
            if done:
                 obs = env.reset()
                 print(f"Episode finished. Total Reward: {total_reward}")
                 total_reward = 0
                 
    except KeyboardInterrupt:
        print("Stopped by user.")
    
    env.close()

if __name__ == "__main__":
    play("mario_ppo_final")
