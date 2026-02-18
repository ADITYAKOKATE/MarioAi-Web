import os
import shutil
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from mario_env import MarioEnv

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env():
    def _init():
        env = MarioEnv()
        return env
    return _init

def train():
    # 0. Clean start
    if os.path.exists("mario_ppo_final.zip"):
        print("Note: Overwriting previous model and starting fresh.")
    
    # 1. Setup Environment
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)

    # 2. Setup Model
    callback = CheckpointCallback(save_freq=20000, save_path=CHECKPOINT_DIR, name_prefix='mario_model')
    
    # Increased total_timesteps to 200,000 for better convergence
    # Increased ent_coef to 0.05 to encourage exploring "Jump" action
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, 
                learning_rate=0.0003, 
                n_steps=2048, 
                batch_size=64, 
                n_epochs=10, 
                gamma=0.99, 
                gae_lambda=0.95, 
                clip_range=0.2, 
                ent_coef=0.05) # <-- Higher entropy to stop getting stuck

    # 3. Train
    print("Starting training with OPTIMIZED settings...")
    try:
        model.learn(total_timesteps=200000, callback=callback)
        print("Training finished.")
        model.save("mario_ppo_final")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        model.save("mario_ppo_interrupted")
        print("Saved 'mario_ppo_interrupted.zip'.")
    except Exception as e:
        print(f"Training failed: {e}")
        model.save("mario_ppo_interrupted")

if __name__ == "__main__":
    train()
