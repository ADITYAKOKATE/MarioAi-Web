import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time

def main():
    try:
        # Create the environment
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        # Restrict the action space to simple movements
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        done = True
        for step in range(500):
            if done:
                state = env.reset()
            
            # Take a random action
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            
            # Render the game
            env.render()
            time.sleep(0.01)

        env.close()
        print("Verification successful: Environment loaded and random agent played.")
    except Exception as e:
        print(f"Verification failed: {e}")

if __name__ == "__main__":
    main()
