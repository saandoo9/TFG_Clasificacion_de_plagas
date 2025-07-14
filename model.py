import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchsummary
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, classification_report
from tqdm.notebook import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import shutil
import zipfile
import os

class InsectClassifier(nn.Module):
    def __init__(self):
        super(InsectClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # Ajuste de dimensiones
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = self.features(x)
        z = z.view(z.size(0), -1)
        return self.classifier(z)
    

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cuda", criterion=nn.BCEWithLogitsLoss):
    
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Evaluación en validación
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []


        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average="macro")  # F1-score macro para clases balanceadas
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Aplicar el scheduler basado en la pérdida de validación
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f}")

        # Guardar el mejor modelo basado en la precisión de validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()

    print(f"Mejor Precisión de Validación: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)  # Restaurar el mejor modelo

    return history

def train_with_two_lr(model, train_loader, val_loader, epochs=20, lr_backbone=0.00001, lr_classifier=0.0001, device="cuda", criterion=nn.BCEWithLogitsLoss):

    model.to(device)
    model.train()
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': lr_backbone},           
        {'params': model.classifier.parameters(), 'lr': lr_classifier}         
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Evaluación en validación
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        all_preds, all_labels = [], []


        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()
            _, preds = torch.max(outputs, 1)

            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_f1 = f1_score(all_labels, all_preds, average="macro")  # F1-score macro para clases balanceadas
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Aplicar el scheduler basado en la pérdida de validación
        scheduler.step(val_loss)
        

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} | "
              )

        # Guardar el mejor modelo basado en la precisión de validación
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = model.state_dict()

    print(f"Mejor Precisión de Validación: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)  # Restaurar el mejor modelo

    return history

def classifier(in_features, num_classes):    # Clasificador para los modelos preentrenados
    cl = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    return cl

