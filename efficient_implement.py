# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 01:49:34 2023

@author: nisar
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import time
from torchsummary import summary
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqtgraph.Qt import QtCore
from IPython import display
import os
import random
import argparse
import matplotlib
from inception_net import GoogLeNet
from sklearn.manifold import TSNE
from background_remove import background_clutter_estimation
#matplotlib.use('Qt5Agg')
from resnet16 import ResNet14
from scipy.spatial import ConvexHull
from torchviz import make_dot
from graphviz import Source
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nice_cm import cm_analysis
from linearregression import LinearRegressor
from lenet import LeNet
from augment_Data import augmentation_data
from tsaug_aug import augment_tag
from combined_data import comb_data
%matplotlib qt

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description='Beam SNR Home Dataset')
    parser.add_argument('--dataset_path', type=str, default="C://Users//nisar//Documents//immercom_session1_session2//data_mmwave_immercom//session2//joint labels kinect and activity session2//",
                       help='The path of the dataset folder')
    parser.add_argument('--num_epochs', type=int, default=150, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='The weight decay regularizer for Adam optimizer')
    parser.add_argument('--beta1', type=float, default=0.9, help='The beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='The beta2 for Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-10, help='The epsilon for Adam optimizer')
    parser.add_argument('--patience', type=int, default=25,
                        help='The patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint_new.pth",
                        help='The path of the checkpoint file to load the model and optimizer state dicts')
    parser.add_argument('--background_remove', type=bool, default=False,
                       help='Enable background subtraction')
    parser.add_argument('--background_path', type=str, default="C://Users//nisar//Documents//data_mehran_GAN//session 1//background subtract//",
                        help='provide files to do background subtraction')
    args = parser.parse_args()
    return args

def train_test_model(net, train_data_loader,  test_data_loader,optimizer, criterion, scheduler1, num_epochs, device,checkpoint):
    best_test_accuracy=0.0
    train_loss = []
    train_acc = []
    test_losses = []
    labels_epoch10=[]
    labels_epochfinal=[]
    embeddings_epoch10=[]
    embeddings_final=[]
    labels_predicted=[]
    
    
    for epoch in range(num_epochs):
        sigma=torch.tensor(random.uniform(0, 40)).to(device)
        print(sigma)
        net.train()
        running_loss = 0.0
        for (samples, labels) in tqdm(train_data_loader):
            samples = samples.to(torch.float32)
            noise = sigma * torch.randn(samples.size(), device=device)
            #samples=samples+noise
            samples = Variable(samples.to(device))
            labels = labels.squeeze()
            labels = Variable(labels.to(device))

            optimizer.zero_grad()

            predict_output = net(samples)
            loss = criterion(predict_output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss.append(running_loss / len(train_data_loader))
        scheduler1.step(train_loss[-1])

        print(f"\nTrain loss for epoch {epoch+1} is {train_loss[-1]:.3f}")

        net.eval()
        correct_train = 0
        loss_train = 0
       
        for i, (samples, labels) in enumerate(train_data_loader):
            with torch.no_grad():
                samples = Variable(samples.to(device))
                labels = labels.squeeze()
                labels = Variable(labels.to(device))
                predict_output = net(samples)
                s=predict_output
                k=labels
                # if epoch==5:
                #     labels_epoch10.append(labels)
                #     embeddings_epoch10.append(predict_output)
                # if epoch==num_epochs-1:
                #     labels_epochfinal.append(labels)
                #     embeddings_final.append(predict_output)
                
                predicted_label = torch.max(predict_output, 1)[1]
                correct_train += (predicted_label == labels).sum().item()
                loss = criterion(predict_output, labels)
                loss_train += loss.item()

        train_acc.append(100 * float(correct_train) / len(train_data_loader.dataset))
        print("Training accuracy:", train_acc[-1])
        #print("Training accuracy:", train_acc[-1])

      
        test_loss = 0
        correct_test = 0
        for i, (samples, labels) in enumerate(test_data_loader):
            with torch.no_grad():
                samples = Variable(samples.to(device))
                labels = labels.squeeze()
                labels = Variable(labels.to(device))
                predict_output = net(samples)
                if epoch==10:
                    labels_epoch10.append(labels)
                    embeddings_epoch10.append(predict_output)
                if epoch==num_epochs-1:
                    labels_epochfinal.append(labels)
                    embeddings_final.append(predict_output)
                predicted_label = torch.max(predict_output, 1)[1]
                if epoch==num_epochs-1:
                    labels_predicted.append(predicted_label)
               
                correct_test += (predicted_label == labels).sum().item()
                loss = criterion(predict_output, labels)
                test_loss += loss.item()
        test_loss /= len(test_data_loader)
        test_losses.append(test_loss)
        test_acc = 100 * float(correct_test) / len(test_data_loader.dataset)
        if  test_acc > best_test_accuracy:
             # Save the model state
             print(".....saving model...")
             best_test_accuracy = test_acc
             torch.save(checkpoint, args.checkpoint_path)
             print("best test accuracy is",best_test_accuracy)
        print("Test accuracy:", test_acc)
        print(f"Test loss is {test_loss:.3f}")
        plt.clf()
        plt.plot(train_loss, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Train/Test Loss Plot', fontsize=20)
        plt.legend()
        plt.pause(0.001)
       

    return labels_predicted,labels_epoch10,labels_epochfinal,embeddings_epoch10,embeddings_final,k,s,predict_output,labels,best_test_accuracy,train_loss,test_losses, test_acc
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Parse arguments
    
    args = parse_args()

    # Set the dataset paths
    data_path = args.dataset_path + 'data_mm.pth'
    labels_path = args.dataset_path + 'labels.pth'
    # sdata=torch.load("C://Users//nisar//Documents//immercom_session1_session2//data_mmwave_immercom//session 1//joint labels kinect and activity session1//data_mm.pth")
    # slabels=torch.load("C://Users//nisar//Documents//immercom_session1_session2//data_mmwave_immercom//session 1//joint labels kinect and activity session1//labels.pth")
    # slabels=slabels.to(torch.int64)    
    # sdata=sdata.to(torch.float32).to(device)
    # Load the dataset
    data = torch.load(data_path)
    labels = torch.load(labels_path)
    labels=labels.to(torch.int64)
    if args.background_remove:
        print("will do background subtraction")
        background_clutter,data=background_clutter_estimation(args.background_path, data)
    data=data.to(torch.float32).to(device)
#    data=data[0:1171,:,:]
 #   labels=labels[0:1171]
   # data=data.unsqueeze(dim=1)
    #data_aug=augmentation_data(data)
    #data_aug=augment_tag(data).to(device)
    #data_aug=data_aug.squeeze().to(device)
    #data=torch.concat((data,data_aug),dim=0)
    #labels=torch.concat((labels,labels),dim=0)
    data=torch.transpose(data,2,1)
    # sdata=torch.transpose(sdata,2,1)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.86, random_state=seed)
    # X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(sdata, slabels, test_size=0.25, random_state=seed)
    # X_train, X_test, y_train, y_test=comb_data()
    # X_train=X_train.to(device).to(torch.float32)
    # X_test=X_test.to(device).to(torch.float32)
    # y_train=y_train.to(device)
    # y_test=y_test.to(device)
    # scaler = StandardScaler()
    # X_train = torch.tensor(scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape),dtype=torch.float32)
    # X_test = torch.tensor(scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape),dtype=torch.float32)
    # # Define model architecture
    ####                           Dataloader
    # train_dataset = TensorDataset(torch.concat((X_train,X_train_s),dim=0), torch.concat((y_train,y_train_s),dim=0))
    train_dataset = TensorDataset(X_train,y_train)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    test_data_loader=source_test_data_loader
    num_train_instances=len(X_train)
    num_test_instances=len(X_test)
    print("n size of training data is", len(X_train))
    print("\n size of test data is", len(X_test))
   # checkpoint_load=torch.load("checkpoint_home_dataset_ADAM_new_forpaper_94.2.pth")
    #net = ResNet14().to(device).to(dtype=torch.float32)
    net=GoogLeNet(num_classes=8).to(device).to(dtype=torch.float32)
    #net=LinearRegressor().to(device).to(dtype=torch.float32)
    #net=LeNet().to(device).to(dtype=torch.float32)
    #net.load_state_dict(checkpoint_load['model_state_dict'])
#     graph = make_dot(net(data), params=dict(net.named_parameters()))
    summary(net,input_size=(30,50))
# # Filter the nodes based on layer type
#     graph.body = [node for node in graph.body if isinstance(node, str) and ('Conv' in node or 'inception' in node)]

# # Display the graph
#     graph.view()
    # Display the graph in Jupyter Notebook or as an image
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,weight_decay=args.weight_decay)
    
    #optimizer.load_state_dict(checkpoint_load['optimizer_state_dict'])
    scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience, verbose=True)
    checkpoint = {'model_state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  }
    labels_predicted,labels_epoch10,labels_epochfinal,embeddings_epoch10,embeddings_final,k,s,predict_output,labels,best_test_accuracy,train_loss,test_losses,test_acc=train_test_model(net, train_data_loader,  test_data_loader,optimizer, criterion, scheduler1, args.num_epochs, device,checkpoint)
    print("best test accuracy is",best_test_accuracy)
    embeddings = s.detach().cpu().numpy()
    labels_epoch_final=torch.concatenate(labels_epochfinal)
    labels_epoch_final_numpy=labels_epoch_final.cpu().numpy()
    embeddings_final_con=torch.concatenate(embeddings_final)
    embeddings_numpy_final=embeddings_final_con.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42,perplexity=5,learning_rate=10,early_exaggeration=10)
    embeddings_tsne = tsne.fit_transform(embeddings_numpy_final)
    label_mapping = {
    0: "AU",
    1: "AW",
    2: "E",
    3: "LHO",
    4: "LL",
    5: "P",
    6: "RHO",
    7: "RL"
    }
    plt.figure(facecolor='white')  # Set the background color to white
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    ax = plt.gca()

# Set the color of the plot area
    ax.set_facecolor('white') 
    plt.title('Train/Test Loss', fontsize=20)
    plt.grid(visible=True)  # Remove the grid lines
    plt.xticks([0, 50, 100, 150])


    plt.legend()
    plt.savefig("four liner layers//heng//train_test_loss_final_inception_net_5e-2_removenoise.jpg", dpi=500)  # Adjust the filename and dpi value as desired
    labels_predicted_final=torch.concatenate(labels_predicted)
    labels_predicted=labels_predicted_final.cpu().numpy()
    labels_ground_truth=labels_epoch_final_numpy
    label_mapping = {
    0: "AU",
    1: "AW",
    2: "E",
    3: "LHO",
    4: "LL",
    5: "P",
    6: "RHO",
    7: "RL"
    }
    cm = confusion_matrix(labels_ground_truth, labels_predicted)

# Get the labels for the confusion matrix
    cm_labels = [label_mapping[i] for i in range(len(label_mapping))]
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=cm_labels, yticklabels=cm_labels)
    plt.xlabel('Predicted Labels',fontsize=14)
    plt.ylabel('True Labels',fontsize=14)
    plt.title('Confusion Matrix',fontsize=20)
    plt.savefig("confusion matrix.png", dpi=1000)  # Adjust the filename and dpi value as desired
    plt.show()
    accuracy = np.trace(cm) / np.sum(cm)
    
    cm_analysis(labels_ground_truth, labels_predicted, "nice_cm.jpeg", [0,1,2,3,4,5,6,7],["AU","AW","E","LHO","LL","P","RHO","RL"] )

# Print the accuracy
    print("Accuracy:", accuracy)
#     plt.figure(figsize=(15, 10))

# # Plot the background clutter
#     plt.plot(data[0].cpu())
    
#     # Set the title
#     plt.title("Sample data", fontsize=20)
#     plt.xlabel("Sample time (s)", fontsize=14)
#     plt.ylabel("Amplitude", fontsize=14)
    
#     # Set the x-axis tick locations and labels
#     tick_locations = [5, 10, 15, 20, 25]
#     tick_labels = ['5', '10', '15', '20', '25']
#     plt.xticks(tick_locations, tick_labels)

# # Save the figure with DPI 400
#     plt.savefig("sample_Data.png", dpi=100)

#     # Show the plot
#     plt.show()



# # Set the background color

# # Save the figure with DPI 400
#     plt.savefig("background_clutter.png", dpi=100)

    

# Convert the numeric labels to corresponding labels
    # labels_mapped = np.array([label_mapping[label] for label in labels_epoch_final_numpy])
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(x=embeddings_tsne[:, 0], y=embeddings_tsne[:, 1], hue=labels_mapped, palette='viridis', alpha=0.8, legend='full')
    # sns.despine()
    # plt.title("T-SNE Embeddings", fontsize=20)  # Adjust the fontsize as desired
    # plt.xlabel("Feature 1", fontsize=14)  # Adjust the fontsize as desired
    # plt.ylabel("Feature 2", fontsize=14)  # Adjust the fontsize as desired
    # plt.savefig("plot_epochfinal_test.png", dpi=400)  # Adjust the filename and dpi value as desired
    # plt.show()
    # k=k.cpu()
# Draw lines to separate different cluster
        