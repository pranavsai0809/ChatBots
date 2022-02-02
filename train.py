import json 
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChatDataset
from hyperparameters import*
from model import NeuralNet

with open('intents.json', 'r') as f:
	intents = json.load(f) #Loads the entire json file as a dictionary 

all_words = []
tags = []
xy = []

for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		all_words.extend(w)
		xy.append((w,tag))

ignore_words = ["?","!",".",","]
all_words = [stem(w.lower()) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # Remove duplicate words
tags = sorted(tags) # Dont need to make a set as its worthless

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence,all_words)
	X_train.append(bag)


	label = tags.index(tag)
	y_train.append(label) # CrossEntropyLoss so only label

X_train = np.array(X_train)
y_train = np.array(y_train)

output_size = len(tags)
input_size = len(X_train[0])

dataset = ChatDataset(X_train,y_train)
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size)

# Loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# Train the model
for epoch in range(num_epochs):
	for(words,labels) in train_loader:
		words = words.to(device)
		labels = labels.to(dtype = torch.long).to(device)

		# Forward pass
		outputs = model(words)
		loss = criterion(outputs,labels)

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch + 1)%100 == 0:
		print(f'Epoch [{epoch+1}/{num_epochs}], Loss : {loss.item():.4f}' )

print(f'Final loss : {loss.item():.4f}')

data = {

		"model_state": model.state_dict(),
		"input_size": input_size,
		"hidden_size": hidden_size,
		"output_size": output_size,
		"all_words": all_words,
		"tags": tags
		
		}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')