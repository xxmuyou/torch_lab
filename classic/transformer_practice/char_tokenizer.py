import torch
from torch.utils.data import DataLoader
from typing import Tuple


class CharacterTokenizer:
    """Simple character-level tokenizer"""

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def build_vocab(self, text):
        """Build vocabulary from text"""
        # Get unique characters
        unique_chars = sorted(list(set(text)))

        # Add special tokens
        special_tokens = ["<pad>", "<unk>", "<start>", "<end>"]
        vocab = special_tokens + unique_chars

        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        self.vocab_size = len(vocab)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {unique_chars[:20]}")

    def encode(self, text):
        """Convert text to token ids"""
        return [self.char_to_idx.get(char, self.char_to_idx["<unk>"]) for char in text]

    def decode(self, token_ids):
        """Convert token ids back to text"""
        return "".join(
            [self.idx_to_char[idx] for idx in token_ids if idx in self.idx_to_char]
        )


def get_dataset(
    text, seq_length: int = 50, batch_size: int = 32, device: str = None
) -> Tuple[DataLoader, CharacterTokenizer]:
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)
    inputs, targets = _sequential_dataset(text, tokenizer, sequence_length=seq_length)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    input_tensors = torch.tensor(inputs).to(device)
    target_tensors = torch.tensor(targets).to(device)

    dataset = torch.utils.data.TensorDataset(input_tensors, target_tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), tokenizer


def _sequential_dataset(text, tokenizer: CharacterTokenizer, sequence_length: int = 50):
    token_ids = tokenizer.encode(text)

    inputs = []
    targets = []
    # Create sequences
    for i in range(len(token_ids) - sequence_length):
        # Input: 50 characters
        input_seq = token_ids[i : i + sequence_length]
        # Target: the next 50 characters (shifted by 1)
        target_seq = token_ids[
            i + sequence_length
        ]  # token_ids[i + 1:i + sequence_length + 1]

        inputs.append(input_seq)
        targets.append(target_seq)

    return inputs, targets


if __name__ == "__main__":
    from read_text import clean_text, extract_text

    book_path = "data/三國演義.epub"
    text = extract_text(book_path)
    text = clean_text(text)
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)

    # Assume you have text: "却说天下大势，分久必合"
    sentence = "却说天下大势，分久必合"
    # Your tokenizer (without special tokens for now)
    inputs, targets = _sequential_dataset(sentence, tokenizer, sequence_length=5)
    print(inputs)
    print(targets)
