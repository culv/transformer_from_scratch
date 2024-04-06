from typing import Optional

import pandas as pd
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        x_column: str,
        y_column: str,
        encoding: Optional[str] = "utf8",
    ):
        """todo: docstring"""
        self.df = pd.read_csv(filepath, encoding=encoding)
        self.x_column = x_column
        self.y_column = y_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.loc[idx, self.x_column]
        y = self.df.loc[idx, self.y_column]
        return x, y


if __name__ == "__main__":
    data_path = "data/Womens_Clothing_E-Commerce_Reviews_CLEANED.csv"
    ds = TextClassificationDataset(
        data_path, x_column="review", y_column="Recommended IND"
    )
    print(ds[0])
