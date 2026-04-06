from typing import List, Tuple, Any
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.data.custom_data import AudioData


class Loader:
    """
    Utility class for splitting a dataset into train, test, and validation sets
    and creating corresponding PyTorch DataLoaders.

    Input:
        - data: List of (spectrogram, label) tuples
            - spectrogram: array-like (e.g., numpy array)
            - label: str, either "HC" or "PD"
    """

    def __init__(
        self,
        data: List[Tuple[Any, str]],
        train_ratio: float,
        test_ratio: float,
        batch_size: int,
    ) -> None:
        """
        Initialize the Loader.

        Args:
            data (List[Tuple[Any, str]]): Dataset with (spectrogram, label) pairs.
            train_ratio (float): Fraction of data used for training.
            test_ratio (float): Fraction of data used for testing (validation gets the remainder).
            batch_size (int): Batch size for DataLoaders.
        """
        self.data: List[Tuple[Any, str]] = data
        self.train_ratio: float = train_ratio
        self.test_ratio: float = test_ratio
        self.batch_size: int = batch_size

        self.train_data, self.test_data, self.val_data = self.split_data()

    def split_data(
        self,
    ) -> Tuple[
        List[Tuple[Any, str]],
        List[Tuple[Any, str]],
        List[Tuple[Any, str]],
    ]:
        """
        Split the dataset into train, test, and validation sets using stratified sampling.

        Returns:
            Tuple containing:
                - train_data: List of (spectrogram, label) for training
                - test_data: List of (spectrogram, label) for testing
                - val_data: List of (spectrogram, label) for validation
        """
        X: List[Tuple[Any, str]] = self.data
        y: List[str] = [label for _, label in self.data]

        # Train vs (Test + Val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            train_size=self.train_ratio,
            stratify=y,
            random_state=42,
            shuffle=True,
        )

        # Test vs Val
        test_size_adjusted: float = self.test_ratio / (1 - self.train_ratio)

        X_test, X_val, y_test, y_val = train_test_split(
            X_temp,
            y_temp,
            train_size=test_size_adjusted,
            stratify=y_temp,
            random_state=42,
            shuffle=True,
        )

        return X_train, X_test, X_val

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, test, and validation sets.

        Returns:
            Tuple containing:
                - train_loader: DataLoader for training set
                - test_loader: DataLoader for testing set
                - val_loader: DataLoader for validation set
        """
        train_dataset = AudioData(self.train_data)
        test_dataset = AudioData(self.test_data)
        val_dataset = AudioData(self.val_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return train_loader, test_loader, val_loader
