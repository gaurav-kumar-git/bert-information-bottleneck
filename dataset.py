from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import config

def get_dataloaders():
    """
    Loads and preprocesses the dataset from a local path.
    This version includes a crucial step to filter out any corrupted
    data points where the label is missing (None).
    """
    print(f"Loading tokenizer from: {config.BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH, local_files_only=True)

    print(f"Attempting to load dataset from local path: {config.DATASET_PATH}")
    dataset = load_dataset(config.DATASET_PATH)

    print("Filtering out examples with missing labels...")
    original_sizes = {split: len(dataset[split]) for split in dataset.keys()}
    
    dataset = dataset.filter(lambda example: example['label'] is not None)
    
    filtered_sizes = {split: len(dataset[split]) for split in dataset.keys()}
    
    for split in original_sizes.keys():
        removed_count = original_sizes[split] - filtered_sizes[split]
        if removed_count > 0:
            print(f"Removed {removed_count} corrupted examples from '{split}' split.")
    
    def tokenize_function(examples):
        return tokenizer(examples["utterance"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    tokenized_datasets = tokenized_datasets.remove_columns(["utterance"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=config.BATCH_SIZE
    )
    
    validation_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        batch_size=config.BATCH_SIZE
    )

    test_dataloader = DataLoader(
        tokenized_datasets["test"], 
        batch_size=config.BATCH_SIZE
    )

    return train_dataloader, validation_dataloader, test_dataloader