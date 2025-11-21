# Main script

'''
FedAvg aggregates model weights without sharing data:
• Works well with small datasets like MNIST.
• Can be extended to real-world FL scenarios using Flower.
• No GPU required for this lab. Why?

Both the dataset (MNIST) and the model (SimpleCNN) are small, and we only train for a few epochs over 3 clients and 5 rounds. 
The total computation is light enough that a standard CPU can finish in reasonable time, so we don't need the parallel processing power of a GPU.
'''

'''
Expected results of the lab:
• To achieve an accuracy of around 85-90% after 5 rounds. You may vary the number of
training rounds (i.e., num_rounds) to see the performance.
• To understand the federated learning workflow.
• To simulate federated learning with VS Code, Python, and PyTorch.
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models import SimpleCNN
from utils import fed_avg

# Config
epochs = 2
num_clients = 3
num_rounds = 5
lr = 0.01

# Data
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
client_data = torch.utils.data.random_split(dataset, [len(dataset)//num_clients]*num_clients)

# Evaluation
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Initialise global model
global_model = SimpleCNN()

# Initialise the training accuracy
round_accuracies = []
for round in range(num_rounds):
    local_weights = []
    for client_idx in range(num_clients):
        model = SimpleCNN()
        model.load_state_dict(global_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loader = DataLoader(client_data[client_idx], batch_size=32, shuffle=True)
        
        # local model training
        model.train()
        for epoch in range(epochs):
            for data, target in loader:
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.CrossEntropyLoss()(output, target)
                loss.backward()
                optimizer.step()
                
        local_weights.append(model.state_dict())
        
    # Model aggregation at the server
    global_model.load_state_dict(fed_avg(local_weights))
    # After model aggregation
    print(f"Training round {round+1} completed.")
    
    # Compute the model accuracy after the current training round
    correct = 0
    global_model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = global_model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            current_accuracy = 100. * correct / len(test_loader.dataset)
            
    print(f"Round {round+1} Accuracy: {current_accuracy:.2f}%")
    # Appendix the current accuracy to the list
    round_accuracies.append(current_accuracy)
    
# Plot the figure of the global model accuracy vs training round
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_rounds+1), round_accuracies, marker='o', linestyle='-', color='blue')
plt.title('Global Model Accuracy per Training Round')
plt.xlabel('Training Round')
plt.ylabel('Global Model Accuracy (%)')
plt.xticks(range(1, num_rounds+1))
plt.grid(True)
plt.show()