import warnings
import os
import sys
from data_reduction.statistic import srs_selection, prd_selection
from data_reduction.geometric import clc_selection, mms_selection, des_selection
from data_reduction.ranking import phl_selection, nrmd_selection
from data_reduction.wrapper import fes_selection
from data_reduction.representativeness import find_epsilon

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #evitar warnings, info tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import io
import time
import math
from codecarbon import OfflineEmissionsTracker
import shutil
from PIL import Image
import json
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import cKDTree
import argparse
import pandas as pd

from ultralytics import YOLO

posibblesMethods=["NONE","SRS","DES","NRMD","MMS","RKMEANS","PRD","PHL","FES"]
    
def PathsImagesFolder(path):
    paths_images = []
    paths_images_only = []
    new_path = 'yolov8/wheelchair-detection-1/train/imagesTodas'
    if not os.path.exists(new_path):
        new_path = path +'/images'
    else:
        folder_path = path +'/imagesTodas'
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = folder_path + "/" + file
                paths_images.append(image_path)
                paths_images_only.append(file)
    print(f'There are {len(paths_images)} images in the path {path}')
    return paths_images, paths_images_only

def categorize_files(folder_path):
    category = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                path_file = os.path.join(root, file)
                with open(path_file, 'r') as archivo:
                    lines = archivo.readlines()
                    # Category 0: Only one object: Wheelchair
                    if len(lines) == 1 and lines[0].startswith('1'):
                        category.append(0)
                    #Category 1: Only one object: Person
                    elif len(lines) == 1 and lines[0].startswith('0'):
                        category.append(1)
                    # Category 2: Multiple objects combining chairs and people
                    elif len(lines) > 1 and any(line.startswith('0') for line in lines) and any(line.startswith('1') for line in lines):
                        category.append(2)
                    # Category 3: Multiple items, all wheelchairs
                    elif len(lines) > 1 and all(line.startswith('1') for line in lines):
                        category.append(3)
                    # Multiple objects, all people, also category 1
                    elif len(lines) > 1 and all(line.startswith('0') for line in lines):
                        category.append(1) 
    return category

def representative_kmeans(X,category,perc):
    n_classes = np.unique(category).shape[0]
    kmeans = KMeans(n_clusters=n_classes) 
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    indexes = np.arange(0,X.shape[0])
    indexesChosen=[]
    perc = perc
    for i in range(kmeans.n_clusters):  
        cluster_center = kmeans.cluster_centers_[i]
    
        distances = []
        for j, label in enumerate(cluster_labels):
            if label == i:
                dist = np.linalg.norm(X[j] - cluster_center)
                distances.append((j, dist))
    
        distances.sort(key=lambda x: x[1])
        num_representatives = min(int(int(X.shape[0]*perc)/n_classes), len(distances)) 
        indexesChosen.extend([indexes[idx] for idx, _ in distances[:num_representatives]])
    
    return indexesChosen
    
def rkmeans(paths_images,tensor_YOLO,y,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init=time.time()
    indexes = representative_kmeans(tensor_YOLO,y,perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end_epsilon(tensor_YOLO,np.array(y),tensor_YOLO[indexes],np.array(y)[indexes])
    representative_images = [paths_images[index] for index in indexes]
    
    end = time.time()
    elapsed_time = end - init
    print(f"The run time of RKMEANS with a reduction to {perc} was: {elapsed_time} seconds")

    paths_images_RepresentativeKMeans=np.array(representative_images)
    print("We have gone from", tensor_YOLO.shape[0] , " to " , paths_images_RepresentativeKMeans.shape[0])
    return paths_images_RepresentativeKMeans
    
class ImageDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return img, label
    def __len__(self):
        return len(self.labels)

class MiModelo(nn.Module):
    def __init__(self,l):
        super(MiModelo, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 32, 5, 2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5, 2)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, l)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x,dim=1)
        return x


def train_step(train_loader, model, args, criterion, optimizer):
    model = model.to(args.device)
    model.train() 
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        optimizer.zero_grad() 
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def train_model(X,y,model,criterion,optimizer,args):
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    for i in range(args.total_epochs):
        print(f"\rEpoch: {i}", end='', flush=True)
        train_step(train_loader, model, args, criterion, optimizer)

def forgetting_step(model, current_accuracy, forgetting_events, X, y, args):
    model = model.to(args.device)
    model.eval()
    n_y = len(y)
    batch_size = args.batch_size
    with torch.no_grad():
        for i in range(0, int(n_y/batch_size)+1):
            batch_X = X[i*batch_size:i*batch_size+batch_size].to(args.device)
            batch_y = y[i*batch_size:i*batch_size+batch_size].to(args.device)
        
            outputs = model(batch_X.to(args.device))
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == batch_y).tolist()
            for j in range(len(correct)):
                indice = i * batch_size + j
                if indice > n_y:
                    continue
                forgetting_events[indice] += 1 if current_accuracy[indice] > correct[j] else 0
                current_accuracy[indice] = correct[j]
                
def train_fes(X,y,model,criterion,optimizer,args,perc):
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    n_y = len(y)
    current_accuracy = np.zeros(n_y, dtype=np.int32)
    forgetting_events = np.zeros(n_y, dtype=np.int32)
    print("Epochs before reduction:")
    for i in range(args.initial_epochs):
        print(f"\rEpoch {i}", end='', flush=True)
        train_step(train_loader, model, args, criterion, optimizer)
        forgetting_step(model, current_accuracy, forgetting_events, X, y, args)
    X_red, y_red = fes_selection(y,current_accuracy, forgetting_events,perc,args.initial_epochs, X=X)
    return X_red, y_red
    
def fes(paths_images,perc,tensor_YOLO,category,args):
    trainImagesPath = 'yolov8/wheelchair-detection-1/train/imagesTodas'
    trainImages = [os.path.join(trainImagesPath,path) for path in os.listdir(trainImagesPath)]
    trainLabelsPath = 'yolov8/wheelchair-detection-1/train/labels'
    trainLabels = categorize_files(trainLabelsPath)
    numCat = np.unique(trainLabels).shape[0]
    
    tensor = torch.zeros((len(trainImages),3,416,416),dtype=torch.float16)
    i=0
    for path in trainImages:
      img = preprocess_img_yolo(path)
      tensor[i,:,:,:] = img
      i += 1
    parser = argparse.ArgumentParser(description='Arguments for the experiments')

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        metavar='LR',
        help='Learning Rate (default: 0.01)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.5,
        metavar='M',
        help='SGD momentum (default: 0.5)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        metavar='N',
        help='input batch size for training (default: 64)')
    parser.add_argument(
        '--no_dropout', action='store_true', default=False, help='remove dropout')
    parser.add_argument(
        '--dropout_prob',
        type=float,
        default=0.33,
        metavar='M',
        help='Dropout probability (default: 0.33)')
    parser.add_argument(
        '--total_epochs',
        type=int,
        default=15,
        metavar='N',
        help='number of epochs to train (default: 15)')
    parser.add_argument(
        '--initial_epochs',
        type=int,
        default=10,
        metavar='N',
        help='number of epochs to train before reduction (default: 10)')
    parser.add_argument(
        '--reduction_ratio',
        type=float,
        default=0.5,
        metavar='perc',
        help='reduction percentage (default: 0.5)')
    parser.add_argument(
        '--n_iter',
        type=int,
        default=10,
        metavar='N_iter',
        help='number of iterations of the experiment (default: 10)')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to do the computations. Can be cou or cuda (default: cpu)')
    argsFES = parser.parse_args([
    '--learning_rate','0.01',
    '--momentum','0.5',
    '--batch_size','15',
    '--no_dropout',
    '--dropout_prob', '0.33',
    '--total_epochs',str(args.epochFES),
    '--initial_epochs',str(args.epochFES),
    '--reduction_ratio',str(perc),
    '--n_iter','15',
    '--device', 'cuda'
    ])
    model = MiModelo(numCat)
    model = model.to(dtype=torch.float16)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=argsFES.learning_rate)
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init=time.time()
    X_res, y_res= train_fes(tensor,torch.tensor(trainLabels),model,criterion,optimizer,argsFES,perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time was of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor,X_res)
    end_epsilon(tensor_YOLO,np.array(category),tensor_YOLO[indexes],np.array(category)[indexes])
    paths_images_reduced = np.array(paths_images)[indexes]
    print(f'We have gone from {len(paths_images)} samples to {len(paths_images_reduced)} samples with the Forgetting Event Score method and a reduction to {1 - perc}')
    return paths_images_reduced
    
def indexesSelected(full_tensor,reduce_tensor):
    indexes = []
    for i, fila in enumerate(full_tensor):
        for fila_res in reduce_tensor:
            if np.array_equal(fila, fila_res):
                indexes.append(i)
                break
    return indexes

def end_epsilon(X,y,X_res,y_res):
    init = time.time()
    epsilon = find_epsilon(X,y,X_res,y_res)
    end = time.time()
    elapsed_time = end - init
    print(f"The time to calculate epsilon-representativity ({epsilon}) has been of {elapsed_time} seconds")
    return epsilon
    
def srs(paths_images,tensor_YOLO,category,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init = time.time()
    X_res, y_res = srs_selection(tensor_YOLO,np.array(category),perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time of SRS with a reduction of {perc} has been of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor_YOLO,X_res)
    end_epsilon(tensor_YOLO,np.array(category),X_res,y_res)
    paths_images_reduced=np.array(paths_images)[indexes]
    return paths_images_reduced

def des(paths_images,tensor_YOLO,category,perc):
    if perc > 0 and perc > 0.3:
        perc_base = 0.2
    elif perc > 0 and perc < 0.3:
        perc_base = 0.05
    print(f"perc_base = {perc_base}")
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init = time.time()
    X_res, y_res = des_selection(tensor_YOLO,np.array(category),perc,perc_base)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time of DES with a reduction of {perc} has been of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor_YOLO,X_res)
    end_epsilon(tensor_YOLO,np.array(category),X_res,y_res)
    paths_images_reduced=np.array(paths_images)[indexes]
    return paths_images_reduced

def nrmd(paths_images,tensor_YOLO,category,perc,argsNRMD):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init = time.time()
    X_res, y_res = nrmd_selection(tensor_YOLO,np.array(category),perc,argsNRMD.decompositionNRMD)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time of NRMD with a reduction of {perc} has been of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor_YOLO,X_res)
    end_epsilon(tensor_YOLO,np.array(category),X_res,y_res)
    paths_images_reduced=np.array(paths_images)[indexes]
    return paths_images_reduced


def phl(paths_images,tensor_YOLO,category,perc,argsPHL):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init = time.time()
    X_res, y_res = phl_selection(tensor_YOLO,np.array(category),int(argsPHL.topologicalRadiusPHL),perc,argsPHL.scoringVersionPHL,int(argsPHL.dimensionPHL),argsPHL.landmarkPHL)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time of PHL with a reduction of {perc} has been of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor_YOLO,X_res)
    end_epsilon(tensor_YOLO,np.array(category),X_res,y_res)
    paths_images_reduced=np.array(paths_images)[indexes]
    return paths_images_reduced

def mms(paths_images,tensor_YOLO,category,perc):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init = time.time()
    X_res, y_res = mms_selection(tensor_YOLO,np.array(category),perc)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time of MMS with a reduction of {perc} has been of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor_YOLO,X_res)
    end_epsilon(tensor_YOLO,np.array(category),X_res,y_res)
    paths_images_reduced=np.array(paths_images)[indexes]
    return paths_images_reduced

def prd(paths_images,tensor_YOLO,category,perc,argsPRD):
    tracker = OfflineEmissionsTracker(country_iso_code="ESP",log_level="ERROR")
    tracker.start()
    init = time.time()
    X_res, y_res = prd_selection(tensor_YOLO,np.array(category),perc,int(argsPRD.sigmaPRD),argsPRD.optPRD)
    print("Estimated emissions: ", tracker.stop()*1000, " CO2 grams")
    end = time.time()
    elapsed_time = end - init
    print(f"The computing time of PRD with a reduction of {perc} has been of: {elapsed_time} seconds")
    indexes = indexesSelected(tensor_YOLO,X_res)
    end_epsilon(tensor_YOLO,np.array(category),X_res,y_res)
    paths_images_reduced=np.array(paths_images)[indexes]
    return paths_images_reduced

def preprocess_img_yolo(img_path):
  img = image.load_img(img_path)
  x = torchvision.transforms.ToTensor()(img)
  x = torch.unsqueeze(x,0)
  return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetFolder',default="yolov8/wheelchair-detection-1/train", type=str, help='Folder where is located the dataset to reduce')
    parser.add_argument('--name', default='SRS', type=str, help='Reduction method to apply')
    parser.add_argument('--perc', default='0.5', type=float, help="Reduction rate to apply (between 0 and 1)")
    parser.add_argument('--epochFES', default='20', type=int, help="Epoch of FES model train", required=False)
    parser.add_argument('--topologicalRadiusPHL', default='0.25', type=float, help="Topological radius for PHL", required=False)
    parser.add_argument('--scoringVersionPHL', default='restrictedDim', type=str, help="Scoring version for PHL", required=False)
    parser.add_argument('--dimensionPHL', default='1', type=int, help="Dimnesion for PHL", required=False)
    parser.add_argument('--landmarkPHL', default='representative', type=str, help="Landmark type for PHL", required=False)
    parser.add_argument('--decompositionNRMD', default='SVD_python', type=str, help="Decomposition type for NRMD", required=False)
    parser.add_argument('--sigmaPRD', default='3', type=int, help="sigma for PRD", required=False)
    parser.add_argument('--optPRD', default='osqp', type=str, help="opt for PRD", required=False)
    args = parser.parse_args()
    method = args.name
    perc = args.perc
    path_folder = "yolov8/wheelchair-detection-1/train/images"  # Reemplaza con la ruta correcta
    path_new = 'yolov8/wheelchair-detection-1/train/imagesTodas'
    if not os.path.exists(path_new):
      os.rename(path_folder, path_new)
      os.makedirs(path_folder)
    else:
      shutil.rmtree(path_folder)
      os.makedirs(path_folder)
            
    if method not in posibblesMethods:
        raise ValueError("The chosen reduction method(--name) is not among the possible ones: ",posibblesMethods)
    elif method == "NONE":
        print("You have not selected any method, so you are going to train with the complete training set.")
            
        archivos = os.listdir(path_new)
        print("Number Original Files:", len(archivos))
        for archivo in archivos:
            if archivo.endswith(".jpg"):
                path_file = os.path.join(path_new, archivo)
                path_file_new = os.path.join(path_folder, archivo)
                shutil.copy(path_file,path_file_new)
        
        print("The training set has a size of: ", len(os.listdir(path_new)))
        
    else:
        print("Selected Method: ", method)

        if perc < 0 or perc > 1:
            raise ValueError("The rate of reduction(--perc) should be between 0 and 1")
        else:
            print("Selected reduction rate: ", perc)
    
            paths_images, paths_images_only = PathsImagesFolder(args.datasetFolder)
            category = categorize_files(args.datasetFolder)
        
            tensor = torch.zeros(len(paths_images),576) # en yolov8 la salida del backbone no son 768 mapa de caracteristicas, sino 576
        
            # yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5m', verbose=False)
            # backbone = yolov5.model.model.model[0:10]

            yolov8 = YOLO('yolov8m.pt')  # Puedes cambiar a 'yolov8n.pt', 'yolov8s.pt', etc.
            backbone = yolov8.model.model[:10]
            
            i=0
            for path in tqdm(paths_images):
              img = preprocess_img_yolo(path)
              features = backbone(img)
              x = torch.nn.AdaptiveAvgPool2d(1)(features)
              x = torch.squeeze(x)
              tensor[i,:] = x
              i+=1
            
            tensor_YOLO = tensor.numpy()
        
            if method == "RKMEANS":
                paths_images_selected = rkmeans(paths_images_only,tensor_YOLO,category,perc)
            elif method == "SRS":
                paths_images_selected = srs(paths_images_only,tensor_YOLO,category,perc)
            elif method == "DES":
                paths_images_selected = des(paths_images_only,tensor_YOLO,category,perc)
            elif method == "NRMD":
                paths_images_selected = nrmd(paths_images_only,tensor_YOLO,category,perc,args)
            elif method == "PHL":
                paths_images_selected = phl(paths_images_only,tensor_YOLO,category,perc,args)
            elif method == "MMS":
                paths_images_selected = mms(paths_images_only,tensor_YOLO,category,perc)
            elif method == "PRD":
                paths_images_selected = prd(paths_images_only,tensor_YOLO,category,perc,args)
            elif method == "FES":
                paths_images_selected = fes(paths_images_only,perc,tensor_YOLO,category,args)
                
            archivos = os.listdir(path_new)
            for archivo in archivos:
                if archivo.endswith(".jpg") and archivo in paths_images_selected:
                    path_file = os.path.join(path_new, archivo)
                    path_file_new = os.path.join(path_folder, archivo)
                    shutil.copy(path_file,path_file_new)
            
            print("Process completed.")
            print("Number of original files:", len(os.listdir(path_new)))
            print("Files after using", method , " reduction method and a percentage of reduction of ", perc, ": ",  len(os.listdir(path_folder)))

if __name__ == "__main__":
    main()