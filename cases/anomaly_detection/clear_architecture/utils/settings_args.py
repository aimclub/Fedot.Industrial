from pydantic import BaseModel


class SettingsArgs(BaseModel):
    visualize: bool
    print_logs: bool
