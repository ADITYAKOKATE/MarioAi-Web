import traceback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import MarioEnv

def make_env():
    def _init():
        env = MarioEnv(render_mode="rgb_array")
        return env
    return _init

try:
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()
    action = [1]
    
    # Try unpacking 5 values (how I modified it)
    print("Testing 5 values unpacking...")
    obs, reward, terminated, truncated, info = env.step(action)
    print("Success 5 values")
except Exception as e:
    print(f"Exception for 5 values: {e}")
    traceback.print_exc()

try:
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    obs = env.reset()
    action = [1]
    
    # Try unpacking 4 values (the original web_app.py)
    print("\nTesting 4 values unpacking...")
    obs, reward, done, info = env.step(action)
    print("Success 4 values")
except Exception as e:
    print(f"Exception for 4 values: {e}")
    traceback.print_exc()
