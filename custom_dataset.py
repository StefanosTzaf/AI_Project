import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class PerImageAnnotationDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, annotation_ext=".txt"):
        """
        Args:
            image_dir (str): Path to image files.
            annotation_dir (str): Path to per-image annotation files.
            transform (callable, optional): Image transforms (should include ToTensor()).
            annotation_ext (str): Annotation file extension (e.g., ".txt").
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform or T.ToTensor()
        self.annotation_ext = annotation_ext

        self.image_filenames = [f for f in os.listdir(image_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    idx2label = {
      
        0:'Portable_Charger_1',
        1:'Portable_Charger_2',
        2:'Mobile_Phone',
        3:'Laptop',
        4:'Tablet',
        5:'Cosmetic',
        6:'Water',
        7:'Nonmetallic_Lighter'
    }
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, image_name)
        annotation_name = os.path.splitext(image_name)[0] + self.annotation_ext
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # Returns a torch.FloatTensor: [C, H, W]

        # Load bounding boxes and class labels
        boxes = []
        labels = []
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip invalid lines
                x1, y1, x2, y2, class_id = map(float, parts)
                boxes.append([x1, y1, x2, y2])
                labels.append(int(class_id))

        target = {
            'bboxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        return image, target, image_name  # Return image, target, and image name
