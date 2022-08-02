from pydantic import BaseModel

class SettingsArgs(BaseModel):
    visualisate: bool
    print_logs: bool
