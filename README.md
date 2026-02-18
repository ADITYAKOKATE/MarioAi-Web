# Mario AI Project (PyGame Edition)

This project uses Reinforcement Learning (PPO algorithm) to train an AI agent to play a custom Super Mario Bros-style game built with PyGame.

## Why this version?
Original attempts to use `gym-super-mario-bros` required complex C++ build tools. This version uses a **custom Python-only environment** that mimics Mario mechanics, making it easy to run on any machine.

## Prerequisites

1.  **System**: Windows (x64), Python 3.8+
2.  **No C++ Build Tools Required!**

## Installation

1.  **Create Virtual Environment** (if not already active):
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    ```powershell
    pip install pygame gymnasium stable-baselines3 torch opencv-python matplotlib shimmy
    ```

## Usage

### 1. Train the Agent
To start training the AI on the custom environment:
```powershell
python train.py
```
- This will start learning and save checkpoints to the `./train/` folder.
- The final model will be saved as `mario_ppo_final.zip`.
- **Note**: Training is faster than the NES emulator version because the custom environment is lightweight.

### 2. Watch the Agent Play
To recall the trained model and see "Red Square Mario" in action:
```powershell
python play.py
```
- Ensure `mario_ppo_final.zip` exists.

### 3. Verify Setup
To check if the environment works:
```powershell
python test_env.py
```

## Project Structure
- `mario_env.py`: The custom Gym environment (Game logic, physics, rendering).
- `train.py`: Main training script using PPO.
- `play.py`: Evaluation/Visualization script.
