import os
import shutil

import torch


def save_model(
        model: torch.nn.Module,
        file_name: str,
        dir_path: str = 'models',
        state_dict: bool = True,
) -> None:
    """Save the model or its state dict.

    Args:
        model: Torch model.
        file_name: File name without extension.
        dir_path: Path to the directory where the model will be saved.
        state_dict: If ``True`` save state_dict with extension ".sd.pt",
            else save all model with extension ".model.pt".
    """
    os.makedirs(dir_path, exist_ok=True)
    file_name = f"{file_name}.{'sd' if state_dict else 'model'}.pt"
    file_path = os.path.join(dir_path, file_name)
    data = model.state_dict() if state_dict else model
    try:
        torch.save(data, file_path)
    except Exception:
        torch.save(model, file_name)
        shutil.move(file_name, dir_path)
    print(f"Model saved to {os.path.abspath(file_path)}.")
