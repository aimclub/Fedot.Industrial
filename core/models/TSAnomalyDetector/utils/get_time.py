from datetime import datetime
"""
    This is getter for formated current time for logging
"""

def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S")


def time_now() -> str:
    now = datetime.now()
    return now.strftime("%H:%M:%S")
