import web_app
import threading
import time

# Mock the start so game_loop runs once
web_app.game_running = True
web_app.game_loop()

# Read the log queue for the error
while not web_app.log_queue.empty():
    log = web_app.log_queue.get()
    if log["type"] == "error":
        print("FOUND ERROR LOG:", log["msg"])
    else:
        print("LOG:", log["msg"])
