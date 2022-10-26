from datetime import datetime


def time_now() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S")