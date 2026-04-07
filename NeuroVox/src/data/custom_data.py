import torch
from typing import List, Tuple, Any
from torch.utils.data import Dataset
from src.constant.constant import labels


class AudioData(Dataset):
    """
    PyTorch Dataset for spectrogram-based audio samples.

    Each item in `data` is a tuple of (spectrogram, label):

        - spectrogram: array-like (e.g., numpy array) of shape (H, W)
        - label: str, one of "HC" (Healthy Control) or "PD" (Parkinson's Disease)

    Functionality:
        - Converts spectrogram to torch.float32 tensor
        - Adds a channel dimension to match CNN input: shape becomes (1, H, W)
        - Encodes labels to integers:
            "HC" -> 0
            "PD" -> 1
    """

    def __init__(self, data: List[Tuple[Any, str]]) -> None:
        """
        Initialize the dataset.

        Args:
            data (List[Tuple[Any, str]]): List of tuples containing
                (spectrogram, label)
        """
        self.data: List[Tuple[Any, str]] = data

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - spectrogram_tensor: torch.float32 tensor of shape (1, H, W)
                - label_tensor: torch.long tensor representing the encoded label
        """
        spec, label = self.data[index][0], self.data[index][1]

        label = labels[label]

        return (
            torch.as_tensor(spec, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(label, dtype=torch.long),
        )
