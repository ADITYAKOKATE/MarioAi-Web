# MUST be the very first thing — before ANY other imports
try:
    from gevent import monkey
    monkey.patch_all(threads=True)
except ImportError:
    pass

import time
import os
import threading
import numpy as np
import cv2

# --- Make pygame headless BEFORE anything else imports it ---
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from queue import Queue, Empty
from flask import Flask, render_template, jsonify, Response

app = Flask(__name__)

# --- Global State ---
game_running = False
frame_queue = Queue(maxsize=300)
log_queue = Queue(maxsize=500)
game_thread = None
goal_reached_flag = False  # Global flag to signal the frontend

def make_env():
    def _init():
        from mario_env import MarioEnv
        env = MarioEnv(render_mode="rgb_array")
        return env
    return _init

def capture_frame(mario_env):
    """Capture a frame from the mario environment as JPEG bytes."""
    import pygame
    view_rect = pygame.Rect(int(mario_env.camera_x), 0, mario_env.window_width, mario_env.window_height)
    viewport = mario_env.surf.subsurface(view_rect)
    view_str = pygame.image.tostring(viewport, "RGB")
    frame = np.frombuffer(view_str, dtype=np.uint8).reshape((mario_env.window_height, mario_env.window_width, 3))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Scale up for better visibility in browser
    frame_bgr = cv2.resize(frame_bgr, (512, 480), interpolation=cv2.INTER_NEAREST)
    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buffer.tobytes()

def push_frame(frame_bytes):
    """Put a frame in the queue, dropping the oldest if full."""
    if frame_queue.full():
        try:
            frame_queue.get_nowait()
        except Empty:
            pass
    frame_queue.put(frame_bytes)

def game_loop():
    global game_running, goal_reached_flag

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    import pygame

    env = None
    try:
        env = DummyVecEnv([make_env()])
        env = VecFrameStack(env, n_stack=4)

        # Load model
        model = None
        for path in ["mario_ppo_final", "mario_ppo_interrupted", "train/mario_model_200000_steps"]:
            try:
                model = PPO.load(path)
                log_queue.put({"type": "info", "msg": f"✅ Model loaded: {path}"})
                break
            except Exception:
                pass

        if model is None:
            log_queue.put({"type": "error", "msg": "❌ Could not find trained model."})
            game_running = False
            return

        log_queue.put({"type": "info", "msg": "--- Starting Smart Play Loop ---"})
        log_queue.put({"type": "info", "msg": "Columns: [Step] | Action | Position | Notes"})

        obs = env.reset()
        total_reward = 0
        step = 0
        stuck_counter = 0
        last_x_pos = 0
        mario = env.envs[0]

        # Capture and push the initial frame
        push_frame(capture_frame(mario))

        while game_running:
            action, _states = model.predict(obs, deterministic=True)

            # --- Smart Heuristics (same as smart_play.py) ---
            current_x = mario.player_pos[0]
            current_y = mario.player_pos[1]

            if abs(current_x - last_x_pos) < 1:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_x_pos = current_x

            heuristic_active = False
            if stuck_counter > 10:
                action = [2]  # FORCE JUMP
                heuristic_active = True
                if stuck_counter > 15:
                    stuck_counter = 0

            if (380 < current_x < 400) or (780 < current_x < 800) or (1280 < current_x < 1300):
                action = [2]
                heuristic_active = True
                stuck_counter = 0

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Capture frame IMMEDIATELY after step - before any delays
            frame = capture_frame(mario)
            push_frame(frame)

            # Log every step (like smart_play.py)
            act_str = ["Stay", "Right", "Jump"][int(action[0])]
            note = "*** FORCE JUMP ***" if heuristic_active else ""
            log_msg = f"[{step:04d}] | {act_str:5} | Pos: ({current_x:.1f}, {current_y:.1f}) {note}"
            log_queue.put({"type": "action" if not heuristic_active else "highlight", "msg": log_msg})

            step += 1
            time.sleep(1 / 30.0)  # ~30fps

            if done:
                final_reward = float(total_reward[0]) if isinstance(total_reward, np.ndarray) else float(total_reward)
                is_goal = info[0].get("is_goal_reached", False)

                log_queue.put({"type": "info", "msg": f"Episode finished. Total Reward: {final_reward:.1f}"})

                if is_goal:
                    log_queue.put({"type": "success", "msg": f"🏆 GOAL REACHED! Final Pos: ({current_x:.1f}, {current_y:.1f})"})
                    goal_reached_flag = True
                    # Keep streaming for 2 seconds so user can see the finish
                    for _ in range(60):
                        if not game_running:
                            break
                        push_frame(capture_frame(mario))
                        time.sleep(1 / 30.0)
                    break
                else:
                    log_queue.put({"type": "error", "msg": "❌ Mario didn't reach the flag. Restarting..."})
                    obs = env.reset()
                    total_reward = 0
                    stuck_counter = 0
                    step = 0
                    log_queue.put({"type": "info", "msg": "-" * 50})

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"GAME LOOP ERROR:\n{tb_str}")
        log_queue.put({"type": "error", "msg": f"Error: {str(e)}"})

    finally:
        if env:
            env.close()
        game_running = False
        log_queue.put({"type": "info", "msg": "Game stopped."})


# --- Blank frame (pre-built once) ---
_blank_frame_bytes = None
def _get_blank_frame():
    global _blank_frame_bytes
    if _blank_frame_bytes is None:
        blank = np.zeros((480, 512, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for game...", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        _, buf = cv2.imencode('.jpg', blank)
        _blank_frame_bytes = buf.tobytes()
    return _blank_frame_bytes


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frame')
def get_frame():
    """Return the latest game frame as a single JPEG.
    The browser polls this endpoint at ~30fps instead of holding
    one permanent streaming connection (which caused worker timeouts).
    """
    try:
        frame_bytes = frame_queue.get_nowait()
    except Empty:
        frame_bytes = _get_blank_frame()

    return Response(
        frame_bytes,
        mimetype='image/jpeg',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
        }
    )

@app.route('/start', methods=['POST'])
def start_game():
    global game_running, game_thread, goal_reached_flag
    if not game_running:
        game_running = True
        goal_reached_flag = False
        while not frame_queue.empty():
            frame_queue.get()
        while not log_queue.empty():
            log_queue.get()

        game_thread = threading.Thread(target=game_loop, daemon=True)
        game_thread.start()
        return jsonify({"status": "started"}), 200
    return jsonify({"status": "already running"}), 200

@app.route('/stop', methods=['POST'])
def stop_game():
    global game_running
    game_running = False
    return jsonify({"status": "stopped"}), 200

@app.route('/status')
def get_status():
    global game_running, goal_reached_flag
    return jsonify({"running": game_running, "goal_reached": goal_reached_flag})

@app.route('/logs')
def get_logs():
    logs = []
    while not log_queue.empty():
        try:
            logs.append(log_queue.get_nowait())
        except Empty:
            break
    return jsonify(logs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
