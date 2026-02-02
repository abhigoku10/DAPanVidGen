import os
import json
import datetime


class Logger:
    def __init__(self, log_dir="logs", log_file="train_log.json", to_console=True):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, log_file)
        self.to_console = to_console
        os.makedirs(log_dir, exist_ok=True)

        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log(self, message: dict):
        entry = {
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **message
        }
        if self.to_console:
            print(f"[{entry['time']}] {entry}")

        with open(self.log_file, "r+") as f:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)

    def log_text(self, text: str):
        entry = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
        if self.to_console:
            print(entry)

        with open(os.path.join(self.log_dir, "train_log.txt"), "a") as f:
            f.write(entry + "\n")
