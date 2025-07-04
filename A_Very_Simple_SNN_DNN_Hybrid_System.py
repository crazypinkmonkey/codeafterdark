from collections import deque
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import nengo
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import nengo_dl
import tensorflow as tf

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5)) 
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

print("Dataset Loaded Successfully")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(8 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)    
        return x
    
model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
print("Training CNN...")
for epoch in range(1):
    running_loss = 0.0
    for images, labels in torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss:.4f}")

print("CNN trained successfully")    

images, labels = next(iter(test_loader))

output = model(images)

_, predicted = torch.max(output.data, 1)

print(f"True Label: {labels.item()}")
print(f"CNN Prediction: {predicted[0].item()}")

cnn_output = output[0].detach().numpy()
cnn_output = cnn_output - cnn_output.min()
cnn_output = cnn_output / cnn_output.max()

modulation = 1.0

if predicted[0].item() != labels.item():
    print("Wrong Prediction - adjusting input signal...")
    modulation = 1.5
else:
    print("Correct Prediction - keeping input signal steady.")     

history_buffer = deque(maxlen=5)     
history_buffer.append(cnn_output)

while len(history_buffer) < 5:
    history_buffer.append(cnn_output) 

def input_func(t):
    avg_input = np.mean(history_buffer, axis=0)
    return modulation * avg_input * np.sin(2 * np.pi * 5 * t)

snn_model = nengo.Network()
with snn_model:
    input_node = nengo.Node(output=input_func)
    snn_layer = nengo.Ensemble(n_neurons=500, dimensions=10)
    nengo.Connection(input_node, snn_layer)
    spike_probe = nengo.Probe(snn_layer.neurons)

with nengo.Simulator(snn_model) as sim:
    sim.run(0.1)

    spike_counts = sim.data[spike_probe].sum(axis=0)

    num_classes = 10
    neurons_per_class = 500//num_classes
    grouped_spikes = [
        spike_counts[i * neurons_per_class: (i + 1) * neurons_per_class].sum()
        for i in range(num_classes)
    ]

    snn_prediction = np.argmax(grouped_spikes)   
    print(f"SNN spike - based prediction: {snn_prediction}") 

plt.figure(figsize=(10, 4))
plt.title("Spiking activity of SNN neurons")
plt.xlabel("Time (s)")
plt.ylabel("Neuron index")
plt.plot(sim.trange(), sim.data[spike_probe])
plt.show()

model.eval()
correct = 0
total = 10
triggered_snn = 0

for i in range(total):
    images, labels = next(iter(test_loader))
    output = model(images)
    _, predicted = torch.max(output.data, 1)

    print(f"\nSample {i+1}:")
    print(f"True Label: {labels.item()}")
    print(f"CNN Prediction: {predicted[0].item()}")

    cnn_output = output[0].detach().numpy()
    cnn_output = cnn_output - cnn_output.min()
    cnn_output = cnn_output / cnn_output.max()

    modulation = 1.0
    if predicted[0].item() != labels.item():
        print("Wrong Prediction - adjusting input signal...")
        modulation = 1.5
        triggered_snn += 1

    history_buffer = deque(maxlen=5)
    history_buffer.append(cnn_output) 

    def input_fuc(t):
        avg_input = np.mean(history_buffer, axis=0)
        return modulation * avg_input * np.sin(2 * np.pi * 5 * t)       
    
    snn_model = nengo.Network()
    with snn_model:
        input_node = nengo.Node(output=input_func)
        snn_layer = nengo.Ensemble(n_neurons=500, dimensions=10)
        nengo.Connection(input_node, snn_layer)
        spike_probe = nengo.Probe(snn_layer.neurons)

    with nengo.Simulator(snn_model) as sim:
        sim.run(0.1)

        spike_counts = sim.data[spike_probe].sum(axis=0)
        neurons_per_class = 500//10

        grouped_spikes = [
            spike_counts[i * neurons_per_class: (i + 1) * neurons_per_class].sum()
            for i in range(10)
        ]         
        snn_prediction = np.argmax(grouped_spikes)
        print(f"SNN spike-based prediction: {snn_prediction}")

        if snn_prediction == labels.item():
            print("SNN got it right!")
        else:
            print("SNN got it wrong")    

