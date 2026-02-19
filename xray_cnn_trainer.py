# xray_cnn_trainer.py
# Copyright (c) 2025 [Mustafa. F 2025 , PA]
#
# All rights reserved.
# This software, including the trained model and associated weights, may not be copied, modified, distributed, or used for commercial purposes without explicit written permission from the author.
#
# Use for research, educational, or demonstration purposes is permitted, provided that proper credit is given.
# Commercial use, redistribution, or integration into products or services requires a paid license or written agreement.
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import io
import zipfile
import json

# --- Streamlit setup ---
st.title("🩻 X-Ray CNN Trainer and Predictor with Folder Uploads")

# --- CNN model definition ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*64*64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# --- Transformations ---
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# --- Session state initialization ---
if 'train_started' not in st.session_state:
    st.session_state.train_started = False
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
if 'predict_files' not in st.session_state:
    st.session_state.predict_files = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'label_map' not in st.session_state:
    st.session_state.label_map = {}
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = None

# --- Dataset class ---
class FolderDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# --- Upload folders for training ---
st.header("Upload Folders for Training")
st.write("Each folder should contain images of a single class. The folder name will be used as the label.")
folder_paths = st.file_uploader("Upload folders as zip files (one folder per class)", type=["zip"], accept_multiple_files=True, key="train_upload")

# Start training only if button clicked and not done yet
if st.button("Train CNN") and not st.session_state.training_done:
    st.session_state.train_started = True

if st.session_state.train_started and not st.session_state.training_done and folder_paths:
    file_paths = []
    labels = []
    label_map = {}
    label_idx = 0

    st.write("Extracting files...")
    progress = st.progress(0)

    for f in folder_paths:
        with zipfile.ZipFile(f) as zip_ref:
            extract_path = f"temp_extract_{f.name}"
            zip_ref.extractall(extract_path)
            folder_name = os.path.basename(os.path.splitext(f.name)[0])
            if folder_name not in label_map:
                label_map[folder_name] = label_idx
                label_idx += 1
            # Collect image paths
            images_in_zip = [os.path.join(root, file)
                             for root, _, files in os.walk(extract_path)
                             for file in files if file.lower().endswith(('png','jpg','jpeg'))]
            file_paths.extend(images_in_zip)
            labels.extend([label_map[folder_name]] * len(images_in_zip))
            progress.progress(min(1.0, len(file_paths)/100))  # crude progress, scales with files

    st.session_state.label_map = label_map
    st.write(f"Classes: {label_map}")

    # Split into training and validation
    dataset = FolderDataset(file_paths, labels, transform=train_transform)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Model, criterion, optimizer
    model = SimpleCNN(num_classes=len(label_map))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    st.write("Training...")
    epochs = 30 # note lower model lower image less epoch, not advised to go over 50 due to overfitting
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0

        total = 0
        model.train()
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += lbls.size(0)
                val_correct += (predicted == lbls).sum().item()
        val_acc = 100 * val_correct / val_total

        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    st.success("Training Complete!")
    st.session_state.model_trained = model
    st.session_state.training_done = True

# --- Download trained model ---
if st.session_state.training_done and st.session_state.model_trained:
    buffer = io.BytesIO()
    payload = {
        "model_state_dict": st.session_state.model_trained.state_dict(),
        "label_map": st.session_state.label_map,
    }
    torch.save(payload, buffer)
    buffer.seek(0)
    st.download_button("Download Trained Model", data=buffer, file_name="xray_cnn.pth")

    if st.session_state.label_map:
        labels_json = json.dumps(st.session_state.label_map, indent=2, sort_keys=True)
        st.download_button("Download Label Map (JSON)", data=labels_json, file_name="labels.json")

# --- Upload trained model for prediction ---
st.header("Upload Trained Model for Prediction")
model_file = st.file_uploader("Upload your trained xray_cnn.pth file", type=["pth"], key="predict_model_upload")
label_map_file = st.file_uploader("Optional: upload label map JSON (e.g. {\"Covid\":0,\"Normal\":1})", type=["json"], key="label_map_upload")

if model_file is not None:
    buffer = io.BytesIO(model_file.read())
    checkpoint = torch.load(buffer, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        checkpoint_label_map = checkpoint.get("label_map", {})
    else:
        state_dict = checkpoint
        checkpoint_label_map = {}
    num_classes = state_dict['classifier.2.weight'].shape[0]
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    st.session_state.model_loaded = True
    st.session_state.model = model
    # Load label map if provided; otherwise keep a sane default for cloud runs
    if label_map_file is not None:
        try:
            label_map_dict = json.load(label_map_file)
            if isinstance(label_map_dict, dict):
                st.session_state.label_map = label_map_dict
            else:
                st.warning("Label map JSON must be an object/dict. Using default class indices.")
                st.session_state.label_map = {}
        except Exception:
            st.warning("Could not read label map JSON. Using default class indices.")
            st.session_state.label_map = {}
    elif checkpoint_label_map and not st.session_state.label_map:
        st.session_state.label_map = checkpoint_label_map
    elif not st.session_state.label_map:
        st.session_state.label_map = {}
    st.success(f"Model loaded with {num_classes} classes! Now upload images to predict.")

# --- Upload images for prediction ---
uploaded_predict = st.file_uploader("Upload images to predict", type=["png","jpg"], accept_multiple_files=True, key="predict_images")
if uploaded_predict:
    st.session_state.predict_files = uploaded_predict
else:
    st.session_state.predict_files = []

# --- Predict button ---
if st.session_state.model_loaded and st.session_state.predict_files:
    if st.button("Predict"):
        model = st.session_state.model
        # Handle either name->idx or idx->name maps; fall back to index labels.
        if st.session_state.label_map:
            keys = list(st.session_state.label_map.keys())
            values = list(st.session_state.label_map.values())
            if all(isinstance(k, int) for k in keys) and all(isinstance(v, str) for v in values):
                reverse_map = st.session_state.label_map
            else:
                reverse_map = {v: k for k, v in st.session_state.label_map.items()}
        else:
            reverse_map = {i: str(i) for i in range(model.classifier[2].out_features)}
        progress = st.progress(0)
        total = len(st.session_state.predict_files)
        for i, f in enumerate(st.session_state.predict_files):
            img = Image.open(f).convert('L')
            img_tensor = val_transform(img).unsqueeze(0)
            output = model(img_tensor)
            pred_class_idx = output.argmax(dim=1).item()
            pred_class = reverse_map.get(pred_class_idx, str(pred_class_idx))
            st.write(f"{f.name} → {pred_class}")
            progress.progress((i+1)/total)

# --- Dependencies ---
st.subheader("Required Dependencies")
st.code("""
streamlit
torch
torchvision
pillow
numpy
""")
