from datetime import datetime


def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S")