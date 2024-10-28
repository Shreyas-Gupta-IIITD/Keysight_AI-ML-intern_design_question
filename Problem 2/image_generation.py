import re
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN  # Clustering algorithm to find nearby shapes

# Function to parse the kicad_pcb file and extract coordinates and layer information
def parse_kicad_pcb(file_content):
    shapes = []
    # Find all (gr_poly ...) patterns with their layer
    gr_poly_patterns = re.findall(r'\(gr_poly\s*\(pts\s*(.*?)\)\s*\(layer (.*?)\)', file_content, re.DOTALL)
    
    # Extract points and layers
    for pattern, layer in gr_poly_patterns:
        points = re.findall(r'\(xy ([\d\-.]+) ([\d\-.]+)\)', pattern)
        shape = [(float(x), float(y)) for x, y in points]
        shapes.append({"points": shape, "layer": layer})
        
    return shapes

# Function to calculate the centroid of a shape
def calculate_centroid(points):
    x_coords, y_coords = zip(*points)
    return np.mean(x_coords), np.mean(y_coords)




# Load the file content
file_path = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Test_Spiral.kicad_pcb'
out_folder = 'image_generation'
Path(out_folder).mkdir(parents=True,exist_ok=True)

with open(file_path, 'r') as f:
    file_content = f.read()

# Parse shapes from the file
shapes = parse_kicad_pcb(file_content)

# Separate shapes by layer dynamically
layered_shapes = {}
for shape in shapes:
    layer = shape["layer"]
    if layer not in layered_shapes:
        layered_shapes[layer] = []  # Add new layer if it doesn't exist
    layered_shapes[layer].append(shape)

# Process each layer independently
for layer, shapes in layered_shapes.items():
    print(f"Processing layer: {layer}")

    # Calculate centroids of each shape in the layer
    centroids = [calculate_centroid(shape["points"]) for shape in shapes]
    centroids = np.array(centroids)

    # Use DBSCAN clustering based on proximity
    clustering = DBSCAN(eps=0.3, min_samples=1).fit(centroids)  # Adjust eps as needed
    num_clusters = len(set(clustering.labels_))
    print(f"Found {num_clusters} clusters in layer {layer}")

    # Save each cluster as a separate image
    for cluster_label in range(num_clusters):
        cluster_shapes = [shape for i, shape in enumerate(shapes) if clustering.labels_[i] == cluster_label]
        
        # Set up the plot for saving
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
        print(f"Saved {output_file}")