import os
os.environ["HF_HUB_OFFLINE"] = "1" 
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import config
from dataset import get_dataloaders
from models import ModelPart1, ModelPart2, ModelPart3
import matplotlib.pyplot as plt
import json 

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    """Trains the model (Part 1) for one epoch."""
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels").to(torch.long)
        
        optimizer.zero_grad()
        logits = model(**batch)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    """Evaluates the model (Part 1) and returns accuracy."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").to(torch.long)
            
            logits = model(**batch)
            
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
    return correct_predictions / total_predictions

def train_one_epoch_reconstruction(model, dataloader, optimizer, loss_fn_c, loss_fn_r, device):
    """Trains the model (Part 2 or 3) for one epoch."""
    model.train()
    total_loss_c, total_loss_r = 0, 0
    for batch in tqdm(dataloader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels").to(torch.long)
        
        optimizer.zero_grad()
        logits, reconstructed, original = model(**batch)
        loss_c = loss_fn_c(logits, labels)
        loss_r = loss_fn_r(reconstructed, original)
        loss = loss_c + config.RECONSTRUCTION_ALPHA * loss_r
        loss.backward()
        optimizer.step()
        
        total_loss_c += loss_c.item()
        total_loss_r += loss_r.item()
        
    avg_loss_c = total_loss_c / len(dataloader)
    avg_loss_r = total_loss_r / len(dataloader)
    return avg_loss_c, avg_loss_r

def evaluate_reconstruction(model, dataloader, device):
    """Evaluates the model (Part 2 or 3) and returns accuracy."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels").to(torch.long)
            
            logits, _, _ = model(**batch)
            
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
    return correct_predictions / total_predictions

def save_results_and_plot(results_data, stage_name):
    """
    Saves the results dictionary to a JSON file and generates a comparison plot.
    
    Args:
        results_data (dict): The dictionary containing the experiment results.
        stage_name (str): The name for the current stage (e.g., 'after_part_1', 'final').
    """
    json_path = 'results/experiment_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"\nUpdated experiment results saved to {json_path}")

    dims = sorted(config.BOTTLENECK_DIMS)
    plt.figure(figsize=(12, 8))
    
    if 'part1' in results_data and results_data['part1']:
        acc_part1 = [results_data['part1'].get(d, 0) for d in dims]
        plt.plot(dims, acc_part1, marker='o', linestyle='-', label='Part 1: Bottleneck Only')

    if 'part2' in results_data and results_data['part2']:
        acc_part2 = [results_data['part2'].get(d, 0) for d in dims]
        plt.plot(dims, acc_part2, marker='s', linestyle='-', label='Part 2: + Reconstruction Loss (AE)')

    if 'part3' in results_data and results_data['part3']:
        acc_part3 = [results_data['part3'].get(d, 0) for d in dims]
        plt.plot(dims, acc_part3, marker='^', linestyle='-', label='Part 3: + Stochasticity (DAE)')

    plt.title('Classification Accuracy vs. Bottleneck Width on CLINC-150', fontsize=16)
    plt.xlabel('Bottleneck Dimension', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.xticks(dims)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=11)
    
    all_accuracies = [acc for part in results_data.values() for acc in part.values()]
    if all_accuracies:
        min_acc = min(all_accuracies)
        plt.ylim(bottom=max(0, min_acc - 0.05), top=1.0)

    plot_path = f'results/comparison_{stage_name}.png'
    plt.savefig(plot_path)
    print(f"Comparison graph saved to {plot_path}")
    plt.close() # Close the figure to free up memory

def main():
    print(f"Using device: {config.DEVICE}")
    
    if not os.path.exists('results'):
        os.makedirs('results')

    train_dl, val_dl, test_dl = get_dataloaders()
    
    results = {'part1': {}, 'part2': {}, 'part3': {}}

    print("\n" + "="*50 + "\n" + " " * 15 + "STARTING PART 1" + "\n" + "="*50)
    for dim in config.BOTTLENECK_DIMS:
        print(f"\n--- Training Part 1 with Bottleneck Dim: {dim} ---")
        model = ModelPart1(bottleneck_dim=dim).to(config.DEVICE)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        best_val_accuracy = 0
        best_model_path = f'results/best_model_part1_dim{dim}.pth'
        
        for epoch in range(config.EPOCHS):
            print(f"Epoch {epoch+1}/{config.EPOCHS}")
            train_loss = train_one_epoch(model, train_dl, optimizer, loss_fn, config.DEVICE)
            val_accuracy = evaluate(model, val_dl, config.DEVICE)
            print(f"Train Loss: {train_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")

        print("Loading best model for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        test_accuracy = evaluate(model, test_dl, config.DEVICE)
        print(f"Final Test Accuracy for dim {dim}: {test_accuracy:.4f}")
        results['part1'][dim] = test_accuracy
    save_results_and_plot(results, 'after_part_1')

    print("\n" + "="*50 + "\n" + " " * 15 + "STARTING PART 2" + "\n" + "="*50)
    for dim in config.BOTTLENECK_DIMS:
        print(f"\n--- Training Part 2 with Bottleneck Dim: {dim} ---")
        model = ModelPart2(bottleneck_dim=dim).to(config.DEVICE)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn_c, loss_fn_r = nn.CrossEntropyLoss(), nn.MSELoss()
        best_val_accuracy = 0
        best_model_path = f'results/best_model_part2_dim{dim}.pth'

        for epoch in range(config.EPOCHS):
            print(f"Epoch {epoch+1}/{config.EPOCHS}")
            loss_c, loss_r = train_one_epoch_reconstruction(model, train_dl, optimizer, loss_fn_c, loss_fn_r, config.DEVICE)
            val_accuracy = evaluate_reconstruction(model, val_dl, config.DEVICE)
            print(f"Train Class. Loss: {loss_c:.4f} | Train Recon. Loss: {loss_r:.4f} | Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")

        print("Loading best model for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        test_accuracy = evaluate_reconstruction(model, test_dl, config.DEVICE)
        print(f"Final Test Accuracy for dim {dim}: {test_accuracy:.4f}")
        results['part2'][dim] = test_accuracy
    save_results_and_plot(results, 'after_part_2')

    print("\n" + "="*50 + "\n" + " " * 15 + "STARTING PART 3" + "\n" + "="*50)
    for dim in config.BOTTLENECK_DIMS:
        print(f"\n--- Training Part 3 with Bottleneck Dim: {dim} ---")
        model = ModelPart3(bottleneck_dim=dim).to(config.DEVICE)
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn_c, loss_fn_r = nn.CrossEntropyLoss(), nn.MSELoss()
        best_val_accuracy = 0
        best_model_path = f'results/best_model_part3_dim{dim}.pth'

        for epoch in range(config.EPOCHS):
            print(f"Epoch {epoch+1}/{config.EPOCHS}")
            loss_c, loss_r = train_one_epoch_reconstruction(model, train_dl, optimizer, loss_fn_c, loss_fn_r, config.DEVICE)
            val_accuracy = evaluate_reconstruction(model, val_dl, config.DEVICE)
            print(f"Train Class. Loss: {loss_c:.4f} | Train Recon. Loss: {loss_r:.4f} | Validation Accuracy: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
        
        print("Loading best model for final test evaluation...")
        model.load_state_dict(torch.load(best_model_path))
        test_accuracy = evaluate_reconstruction(model, test_dl, config.DEVICE)
        print(f"Final Test Accuracy for dim {dim}: {test_accuracy:.4f}")
        results['part3'][dim] = test_accuracy
    save_results_and_plot(results, 'final_all_parts')

    print("\n" + "="*50 + "\n" + " " * 18 + "FINAL RESULTS SUMMARY" + "\n" + "="*50)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()