import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.optim import Adam
import torch.onnx

# alpaca / MLteam0719$$

# load data
df = pd.read_excel(r'./Iris_dataset.xlsx')
print('Take a look at sample from the dataset:')
print(df.head())

print('\nOur dataset is balanced and has the following values to predict:')
print(df['Iris_Type'].value_counts())

# Convert Iris species into numeric types: Iris-setosa=0, Iris-versicolor=1, Iris-virginica=2.  
labels = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2} 
df['IrisType_num'] = df['Iris_Type']   # Create a new column "IrisType_num" 
df.IrisType_num = [labels[item] for item in df.IrisType_num]  # Convert the values to numeric ones 

# Define input and output datasets 
input = df.iloc[:, 1:-2]            # We drop the first column and the two last ones. 
print('\nInput values are:') 
print(input.head())   
output = df.loc[:, 'IrisType_num']   # Output Y is the last column  
print('\nThe output value is:') 
print(output.head())

# Convert Input and Output data to Tensors and create a TensorDataset 
input = torch.Tensor(input.to_numpy())      # Create tensor of type torch.float32 
print('\nInput format: ', input.shape, input.dtype)     # Input format: torch.Size([150, 4]) torch.float32 
output = torch.tensor(output.to_numpy())        # Create tensor type torch.int64  
print('Output format: ', output.shape, output.dtype)  # Output format: torch.Size([150]) torch.int64 
data = TensorDataset(input, output)    # Create a torch.utils.data.TensorDataset object for further data manipulation


# Split to Train, Validate and Test sets using random_split 
train_batch_size = 10        
number_rows = len(input)    # The size of our dataset or the number of rows in excel table.  
#print('\nRow Count: ', number_rows)
test_split = int(number_rows*0.3)  
validate_split = int(number_rows*0.2) 
train_split = number_rows - test_split - validate_split     
#print('\ntrain_split Count: ', train_split)
train_set, validate_set, test_set = random_split( 
    data, [train_split, validate_split, test_split])    
 
# Create Dataloader to read the data within batch sizes and put into memory. 
train_loader = DataLoader(train_set, batch_size = train_batch_size, shuffle = True) 
validate_loader = DataLoader(validate_set, batch_size = 1) 
test_loader = DataLoader(test_set, batch_size = 1)


input_size = list(input.shape)[1]
learning_rate = 0.01
output_size = len(labels)

#print('\n Input.shape: ',input.shape)
#print('\n list(input.shape): ',list(input.shape)[1])

class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()

        self.layer1 = nn.Linear(input_size, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, output_size)


    def forward(self, x):
        x1 = F.relu(self.layer1(x))
        x2 = F.relu(self.layer2(x1))
        x3 = self.layer3(x2)
        return x3

model = Network(input_size, output_size)
print('\n model: ',model)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("the model will be running on", device, "device\n")
model.to(device)

def saveModel():
    path = "./NetModel.pth"
    torch.save(model.state_dict(), path)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

#train func
def train(num_epochs):
    best_accuracy = 0.0

    print("Begin training...")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0

        #train loop
        for data in train_loader:
            inputs, outputs = data
            optimizer.zero_grad()
            prericted_outputs = model(inputs)
            train_loss = loss_fn(prericted_outputs, outputs)
            train_loss.backward()
            optimizer.step()
            running_train_loss += train_loss.item()

        train_loss_value = running_train_loss/len(train_loader)
        #validation loop
        with torch.no_grad():
            model.eval()
            for data in validate_loader:
                inputs, outputs = data
                prericted_outputs = model(inputs)
                val_loss = loss_fn(prericted_outputs, outputs)

                _, predicted = torch.max(prericted_outputs, 1)
                running_vall_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()

        val_loss_value = running_vall_loss/len(validate_loader)

        accuracy = (100 * running_accuracy / total)

        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
        
        print('Completed training batch', epoch, 'Training Loss is: %.4f' %train_loss_value, 'Validation Loss is: %.4f' %val_loss_value, 'Accuracy is %d %%' %(accuracy))

def test():
    model = Network(input_size, output_size)
    path = "NetModel.pth"
    model.load_state_dict(torch.load(path))

    running_accuracy = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            outputs = outputs.to(torch.float32)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()
        print('Accyracy of the model based on the test set of', test_split, 'input is: %d %%' %(100 * running_accuracy / total))

def test_species():
    model = Network(input_size, output_size)
    model.load_state_dict(torch.load("NetModel.pth"))

    labels_length = len(labels)
    labels_correct = list(0. for i in range(labels_length))
    labels_total = list(0. for i in range(labels_length))

    with torch.no_grad():
        for data in test_loader:
            inputs, outputs = data
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs, 1)

            labels_correct_running = (predicted == outputs).squeeze()
            label = outputs[0]
            if labels_correct_running.item():
                labels_correct[label] += 1
            labels_total[label] += 1

    label_list = list(labels.keys())
    for i in range(output_size):
        print('Accuracy to predict %5s : %2d %%' %(label_list[i], 100 * labels_correct[i] / labels_total[i]))

def convert(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "Network.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['output'], # the model's output names 
         dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                                'output' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == "__main__":
    num_epochs = 80
    train(num_epochs)
    print('finnished Training\n')
    test()
    test_species()
    convert()