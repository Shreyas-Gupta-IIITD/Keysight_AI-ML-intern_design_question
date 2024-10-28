import os
import re
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset

model_path = 'vgg_19_10.pth'
out_folder = 'inference'
os.makedirs(out_folder,exist_ok=True)
file_path = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Test_Spiral.kicad_pcb'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


##########################################################################################################
#                                               File Processing
##########################################################################################################

# Define transformation to match training setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to parse kicad_pcb file and extract shapes
def parse_kicad_pcb(file_content):
    shapes = []
    gr_poly_patterns = re.findall(r'\(gr_poly\s*\(pts\s*(.*?)\)\s*\(layer (.*?)\)', file_content, re.DOTALL)
    
    for pattern, layer in gr_poly_patterns:
        points = re.findall(r'\(xy ([\d\-.]+) ([\d\-.]+)\)', pattern)
        shape = [(float(x), float(y)) for x, y in points]
        shapes.append({"points": shape, "layer": layer})
        
    return shapes

# Calculate the centroid of a shape for clustering
def calculate_centroid(points):
    x_coords, y_coords = zip(*points)
    return np.mean(x_coords), np.mean(y_coords)

# Function to process the input file and return transformed images for inference
def prepare_inference_images(file_path,out_folder='inference'):
    # Load file content
    with open(file_path, 'r') as f:
        file_content = f.read()

    # Parse shapes from the file
    shapes = parse_kicad_pcb(file_content)

    # Separate shapes by layer dynamically
    layered_shapes = {}
    for shape in shapes:
        layer = shape["layer"]
        if layer not in layered_shapes:
            layered_shapes[layer] = []
        layered_shapes[layer].append(shape)

    # List to hold transformed images for inference
    inference_images = []

    # Process each layer independently
    for layer, shapes in layered_shapes.items():
        # Calculate centroids for clustering
        centroids = [calculate_centroid(shape["points"]) for shape in shapes]
        centroids = np.array(centroids)

        # Use DBSCAN clustering based on proximity
        clustering = DBSCAN(eps=0.3, min_samples=1).fit(centroids)
        num_clusters = len(set(clustering.labels_))

        # Create an image for each cluster and apply transformations
        for cluster_label in range(num_clusters):
            cluster_shapes = [shape for i, shape in enumerate(shapes) if clustering.labels_[i] == cluster_label]

            # Set up the plot for each cluster
            fig, ax = plt.subplots(figsize=(5, 5))
            for shape in cluster_shapes:
                x, y = zip(*shape["points"])
                ax.plot(x, y, linewidth=0.5)
                ax.fill(x, y, alpha=0.3)
            
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')

            
            # Save each cluster as an image file
            output_file = f"{out_folder}/layer_{layer}cluster{cluster_label}.png"
            fig.savefig(output_file, pad_inches=0, dpi=300)
            plt.close(fig)  # Close the figure to free memory
            
            # # Save plot to in-memory file
            # buf = BytesIO()
            # fig.savefig(buf, format='png', pad_inches=0, dpi=300)
            # plt.close(fig)  # Close plot to free memory

            # # Load the image from buffer, apply transformation
            # buf.seek(0)
            # img = Image.open(buf).convert('RGB')
            # img = transform(img)  # Apply transformations
            # inference_images.append(img)
            # buf.close()


prepare_inference_images(file_path,out_folder=out_folder)

##########################################################################################################
#                                               Data Loader
##########################################################################################################

class SpiralDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.directory = [os.path.join(directory, img) for img in os.listdir(directory)]

    def __len__(self):
        return len(self.directory)

    def __getitem__(self, idx):
        img_path = self.directory[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image,img_path

dataset = SpiralDataset(directory=out_folder, transform=transform)

test_loader = DataLoader(dataset, batch_size=2, shuffle=False,num_workers=4, pin_memory=True)

##########################################################################################################
#                                               Load Model
##########################################################################################################

# Load model
model = models.vgg19()
# model = models.vgg16()
# model = models.alexnet()
model.classifier[6] = torch.nn.Linear(4096, 2)  # Change output layer to match 2 classes

# Load the state_dict
model.load_state_dict(torch.load(model_path,weights_only=True, map_location=device))
model = model.to(device)

##########################################################################################################
#                                               Eval Model
##########################################################################################################

# Evaluate the model
model.eval()
class_0_count = 0  # Counter for predictions of class 0
with torch.no_grad():
    for inputs,paths in test_loader:
        outputs = model(inputs.to(device))
        _, preds = torch.max(outputs, 1)
        for pred, path in zip(preds, paths):
            if pred.item() == 0:  # Check if the predicted class is 0
                print(path)  # Print the path of the image
        # Count predictions of class 0
        class_0_count += (preds == 0).sum().item()

print(f"The model predicted {class_0_count} Spirals.")
