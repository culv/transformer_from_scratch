from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        filepath: str,
        x_column: str,
        y_column: str,
        encoding: Optional[str] = "utf8",
        transform: Optional[Callable] = None,
    ):
        """todo: docstring"""
        self.df = pd.read_csv(filepath, encoding=encoding)
        self.x_column = x_column
        self.y_column = y_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        data = tuple(self.df.loc[idx, [self.x_column, self.y_column]])
        if self.transform:
            data = self.transform(data)
        return data


def tokenize(
    context_length: int,
    tokenizer: Optional[PreTrainedTokenizer] = AutoTokenizer.from_pretrained("gpt2"),
) -> Callable:
    """On-the-fly transform for a dataset. Rather than the dataset returning (text, label), this will
    tokenize the text and return (tokens, mask, label)

    Args:
        context_length: How long the context length is; the tokenizer will pad/truncate to this length
        tokenizer: A PreTrainedTokenizer from Huggingface (default is the BPE tokenizer from GPT2)

    Returns:
        transform: A method to transform incoming data from (text, label) to (tokens, mask, label)
    """
    tokenizer.pad_token = tokenizer.eos_token

    def transform(data: Tuple[str, int]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Given a data-label pair (text, label), tokenize the text and return (tokens, mask, label)"""
        x, y = data
        tokens = tokenizer(
            x,
            max_length=context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens.input_ids, tokens.attention_mask, y

    return transform


if __name__ == "__main__":
    data_path = "data/Womens_Clothing_E-Commerce_Reviews_CLEANED.csv"
    ds = TextClassificationDataset(
        data_path, x_column="review", y_column="Recommended IND"
    )
    print(ds[0])

    ds = TextClassificationDataset(
        data_path,
        x_column="review",
        y_column="Recommended IND",
        transform=tokenize(160),
    )
    print(ds[0])
