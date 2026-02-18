from mario_env import MarioEnv
import time

def main():
    try:
        env = MarioEnv(render_mode="human")
        obs, _ = env.reset()
        print(f"Observation shape: {obs.shape}")
        
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            env.render()
            step += 1
            time.sleep(0.05)
            
            if done:
                print(f"Episode finished after {step} steps. Reward: {reward}")
                break
                
        env.close()
        print("Environment test passed!")
    except Exception as e:
        print(f"Environment test failed: {e}")
        raise e

if __name__ == "__main__":
    main()
