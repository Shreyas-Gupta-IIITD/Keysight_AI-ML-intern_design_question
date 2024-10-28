# import matplotlib.pyplot as plt
# import re
# import matplotlib.cm as cm
# import numpy as np

# # Function to parse the kicad_pcb file and extract coordinates
# def parse_kicad_pcb(file_content):
#     shapes = []
#     # Find all (gr_poly ...) patterns
#     # gr_poly_patterns = re.findall(r'\(gr_poly \(pts (.*?)\)\(layer', file_content, re.DOTALL)
#     # gr_poly_patterns = re.findall(r'\(gr_poly\s*\(pts\s*([^\)]*?)\)\(layer', file_content, re.DOTALL)
#     gr_poly_patterns = re.findall(r'\(gr_poly\s*\(pts\s*\((.*?)\)\s*\)\s*\(layer', file_content, re.DOTALL)


#     print(len(gr_poly_patterns))
#     # Extract points for each pattern
#     for pattern in gr_poly_patterns:
#         points = re.findall(r'\(xy ([\d\-.]+) ([\d\-.]+)\)', pattern)
#         shape = [(float(x), float(y)) for x, y in points]
#         shapes.append(shape)
#     return shapes

# # Function to plot all shapes in a single image with different colors
# def plot_shapes(shapes, output_file):
#     plt.figure(figsize=(8, 8))
    
#     # Generate a color map with a unique color for each shape
#     colors = cm.get_cmap("viridis", len(shapes))  # You can change "viridis" to other color maps if you prefer

#     for i, shape in enumerate(shapes):
#         x, y = zip(*shape)  # Separate x and y coordinates
#         plt.plot(x, y, color=colors(i))  # Plot the shape with a unique color
#         plt.fill(x, y, color=colors(i), alpha=0.3)  # Fill the shape with the same color

#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.axis('off')
    
#     # Save the image
#     plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
#     plt.close()  # Close the figure to free memory
#     print(f"Saved {output_file}")

# # Load the file content
# file_path = '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Akshet/Test_Spiral.kicad_pcb'

# with open(file_path, 'r') as f:
#     file_content = f.read()

# # Parse the kicad_pcb file
# shapes = parse_kicad_pcb(file_content)

# # Plot all shapes in a single image and save it with different colors
# output_file = "kicad_shapes_colored.png"
# plot_shapes(shapes, output_file)



import re
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Function to parse the kicad_pcb file and extract coordinates
def parse_kicad_pcb(file_content):
    shapes = []
    # Find all (gr_poly ...) patterns
    gr_poly_patterns = re.findall(r'\(gr_poly\s*\(pts\s*\((.*?)\)\s*\)\s*\(layer', file_content, re.DOTALL)

    print(f"Number of shapes found: {len(gr_poly_patterns)}")
    # Extract points for each pattern
    for pattern in gr_poly_patterns:
        points = re.findall(r'\(xy ([\d\-.]+) ([\d\-.]+)\)', pattern)
        shape = [(float(x), float(y)) for x, y in points]
        shapes.append(shape)
    return shapes

# Function to plot all shapes in a single image with enhanced styling
def plot_shapes(shapes, output_file):
    plt.figure(figsize=(10, 10))
    
    # Generate a color map with a unique color for each shape
    colors = cm.get_cmap("tab20", len(shapes))  # Changed to "tab20" for distinct colors

    for i, shape in enumerate(shapes):
        x, y = zip(*shape)  # Separate x and y coordinates
        plt.plot(x, y, color=colors(i), linewidth=0.01)  # Thicker line for clearer boundaries
        plt.fill(x, y, color=colors(i), alpha=0.5)  # Fill with moderate transparency

    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    
    # Save the image with a high DPI for better quality
    plt.savefig(output_file, pad_inches=0, dpi=1200)
    plt.close()  # Close the figure to free memory
    print(f"Saved {output_file}")

# Load the file content
file_path = '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Akshet/Test_Spiral.kicad_pcb'

with open(file_path, 'r') as f:
    file_content = f.read()

# Parse the kicad_pcb file
shapes = parse_kicad_pcb(file_content)

# Plot all shapes in a single image and save it with improved style
output_file = "kicad_shapes_enhanced_dpi_1200.png"
plot_shapes(shapes, output_file)
