import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import zipfile
from pathlib import Path
import json
import tarfile

class CocoDataset(Dataset):                   #Clase para almacenar las imágenes con sus anotaciones (formato COCO)
    def __init__(self, image_dir, annotation_file, transform=None):
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        self.annotations = {ann['image_id']: ann['bbox'] for ann in coco_data['annotations']}
        self.image_ids = [img_id for img_id in self.image_id_to_filename if img_id in self.annotations]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.image_id_to_filename[image_id]
        img_path = os.path.join(self.image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        bbox = self.annotations[image_id]
        if self.transform:
            image = self.transform(image)
        return img_path, [bbox]


class CustomImageDataset(Dataset):            #Clase para almacenar las imágenes con sus etiquetas
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Recorrer directorios
        for class_name in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                label = int(class_name.split('_')[-1])  # Obtener etiqueta de la carpeta
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):               #Se aplica la transformación al momento de acceder a la imagen para aplicar aumento de datos
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def no_transform_img(self,idx):           #Devuelve la imagen sin transformar
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        return image

class SquareCropTransform:
    def __call__(self, image):
        width, height = image.size
        min_dim = min(width, height)
        return transforms.functional.center_crop(image, min_dim)


def extract_zip(zip_path, extract_to):     #Extrae archivos .zip
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

def extract_tar(archive_path, destination_path):  #Extrae archivos .tar
    
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    try:
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(path=destination_path)
    except Exception as e:
        print(f"Error al extraer: {e}")


def load_binary_data(data_dir):  #Carga un dataset de clasificación binaria
    data_dir = Path(data_dir)
    image_paths, labels = [], []
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir():
            label = 0 if folder.name==("class_0") else 1
            for img_path in folder.glob("*.*"):
                image_paths.append(img_path)
                labels.append(label)
    return image_paths, labels

def load_data(data_dir, exclude_classes=set()):  #Carga un dataset con múltiples clases
    data_dir = Path(data_dir)
    image_paths, labels = [], []
    label_mapping = {}
    current_label = 0
    for folder in sorted(data_dir.iterdir()):
        if folder.is_dir() and folder.name not in exclude_classes:
            label_mapping[folder.name] = current_label
            for img_path in folder.glob("*.*"):
                image_paths.append(img_path)
                labels.append(current_label)
            current_label += 1
    return image_paths, labels
