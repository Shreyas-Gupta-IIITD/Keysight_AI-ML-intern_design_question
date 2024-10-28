import os
import shutil

def consolidate_files(folder_paths, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Loop through each folder path in the list
    for folder_path in folder_paths:
        # Check if the folder exists
        if not os.path.isdir(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        # Loop through all files in the current folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Move the file to the destination folder
                destination_path = os.path.join(destination_folder, filename)
                shutil.copy(file_path, destination_path)
                # print(f"Moved {file_path} to {destination_path}")

# Example usage:
folders = ["/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Spirals",
           "/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/one_spiral_aug",
           "/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/two_spiral_aug"
           ]
destination = "/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Data_Spiral"
consolidate_files(folders, destination)
print("done")


image_paths = [os.path.join(destination, f) for f in os.listdir(destination)]
print(len(image_paths))
