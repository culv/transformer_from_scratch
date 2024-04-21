from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import PreTrainedTokenizer, AutoTokenizer


class BalancedClassRandomSampler(WeightedRandomSampler):
    """Given imbalanced class labels, randomly oversample minority class/undersample majority
    class so that classes appear balanced"""

    def __init__(self, labels: pd.Series):
        num_classes = len(labels.unique())
        num_samples = len(labels)
        samples_per_class = labels.value_counts()
        weights_per_class = num_samples / (num_classes * samples_per_class)
        weights = labels.replace(weights_per_class).tolist()
        super().__init__(weights=weights, num_samples=num_samples, replacement=True)


class PandasDataset(Dataset):
    """A really general dataset, given a CSV file and the labels for input/output
    columns, it will return the (input, output) for a given index. If input/output
    transforms were given, it'll apply those first and return the result"""

    def __init__(
        self,
        filepath: str,
        input_column: str,
        output_column: str,
        input_transform: Optional[Callable] = None,
        output_transform: Optional[Callable] = None,
        encoding: Optional[str] = "utf8",
    ):
        # todo: for really large CSV files, dont load entire df into memory
        self.df = pd.read_csv(filepath, encoding=encoding)
        self.input_column = input_column
        self.output_column = output_column
        self.input_transform = input_transform
        self.output_transform = output_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        input = self.df.loc[idx][self.input_column]
        output = self.df.loc[idx][self.output_column]

        # todo: handle lists of transforms as well
        if self.input_transform:
            input = self.input_transform(input)

        if self.output_transform:
            output = self.output_transform(output)

        return input, output


Gpt2Tokenizer = AutoTokenizer.from_pretrained("gpt2")


# todo: maybe just add tokenizer to dataset class
def tokenize(
    context_length: int,
    tokenizer: Optional[PreTrainedTokenizer] = Gpt2Tokenizer,
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

    def transform(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given some text, tokenize it and return (token_ids, mask)"""
        output = tokenizer(
            text,
            max_length=context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        token_ids = output.input_ids.squeeze()
        mask = output.attention_mask.squeeze()
        return token_ids, mask

    return transform


if __name__ == "__main__":
    # Testing out a classification dataset (text -> label)
    data_path = "data/Womens_Clothing_E-Commerce_Reviews_CLEANED.csv"
    ds = PandasDataset(
        data_path, input_column="review", output_column="Recommended IND"
    )
    print(ds[0])

    ds = PandasDataset(
        data_path,
        input_column="review",
        output_column="Recommended IND",
        input_transform=tokenize(160),
    )
    print(ds[0])

    # Testing out a translation dataset (text -> text)
    data_path = "data/translation.csv"
    ds = PandasDataset(
        data_path,
        input_column="en",
        output_column="fr",
        input_transform=tokenize(10),
        output_transform=tokenize(10),
    )
    print(ds[0])
