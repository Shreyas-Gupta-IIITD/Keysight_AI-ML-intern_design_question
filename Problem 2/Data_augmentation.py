'''
Create Spirals


# # Use PiecewiseAffine for more pronounced distortion effects
# A.PiecewiseAffine(scale=(0.03, 0.05), p=1),
# A.PiecewiseAffine(scale=(0.05, 0.07), p=1),
# A.PiecewiseAffine(scale=(0.07, 0.1), p=1),

# # Multiple GridDistortion variations
# A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
# A.GridDistortion(num_steps=7, distort_limit=0.4, p=1),
# A.GridDistortion(num_steps=9, distort_limit=0.5, p=1),
# A.GridDistortion(num_steps=6, distort_limit=0.6, p=1),
# A.GridDistortion(num_steps=8, distort_limit=0.7, p=1),
# A.GridDistortion(num_steps=10, distort_limit=0.8, p=1),
# A.GridDistortion(num_steps=12, distort_limit=0.9, p=1),

'''


import cv2
import albumentations as A
import os
import random

# Define a comprehensive list of significant augmentation transformations
augmentation_list = [
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.Rotate(limit=30, p=1),
    A.RandomBrightnessContrast(p=1),
    A.RandomBrightnessContrast(p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1),
    
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=1),
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
    A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
    A.Blur(blur_limit=5, p=1),
    A.ToGray(p=1),
    A.Solarize(threshold=128, p=1),
    
    A.InvertImg(p=1),
    A.RandomFog(fog_coef_lower=0.8, fog_coef_upper=1, alpha_coef=0.5, p=1),
    A.RandomFog(fog_coef_lower=0.8, fog_coef_upper=1, alpha_coef=0.8, p=1),
    A.RandomRain(slant_lower=-20, slant_upper=20, drop_length=15, drop_color=(200, 200, 200), blur_value=3, p=1),
    A.RandomSnow(snow_point_lower=0.2, snow_point_upper=0.4, p=1),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=1),
    A.ToSepia(p=1),
    
    # # Use PiecewiseAffine for more pronounced distortion effects
    # A.PiecewiseAffine(scale=(0.03, 0.05), p=1),
    # A.PiecewiseAffine(scale=(0.05, 0.07), p=1),
    # A.PiecewiseAffine(scale=(0.07, 0.1), p=1),

    # # Multiple GridDistortion variations
    # A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
    # A.GridDistortion(num_steps=7, distort_limit=0.4, p=1),
    # A.GridDistortion(num_steps=9, distort_limit=0.5, p=1),
    # A.GridDistortion(num_steps=6, distort_limit=0.6, p=1),
    # A.GridDistortion(num_steps=8, distort_limit=0.7, p=1),
    # A.GridDistortion(num_steps=10, distort_limit=0.8, p=1),
    # A.GridDistortion(num_steps=12, distort_limit=0.9, p=1),
    
]


# # Function to create the Spirals and Non Spirals
# def augment_and_save_individual(image_paths, output_folder="Non_Spirals"):
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Loop through each image path
#     for idx, image_path in enumerate(image_paths):
#         # Load the image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Could not load image {image_path}. Skipping.")
#             continue

#         # Apply each augmentation in the list separately
#         for aug_idx, augmentation in enumerate(augmentation_list):
#             augmented_image = augmentation(image=image)["image"]
#             # Save the augmented image
#             output_path = os.path.join(output_folder, f"{output_folder}_image_{idx+1}_aug_{aug_idx+1}.jpg")
#             cv2.imwrite(output_path, augmented_image)




# Function to apply each augmentation separately and save the images
# def augment_and_save_individual(image_paths, output_folder="one_non_spiral_aug"):
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Loop through each image path
#     for idx, image_path in enumerate(image_paths):
#         # Load the image
#         image = cv2.imread(image_path)
#         if image is None:
#             print(f"Could not load image {image_path}. Skipping.")
#             continue

#         # Apply each augmentation in the list separately
#         for aug_idx, augmentation in enumerate(augmentation_list):
#             augmented_image = augmentation(image=image)["image"]
#             # Save the augmented image
#             output_path = os.path.join(output_folder, f"{output_folder}_image_{idx+1}_aug_{aug_idx+1}.jpg")
#             cv2.imwrite(output_path, augmented_image)



# Function to apply 3 augmentation separately and save the images
def augment_and_save_individual(image_paths, output_folder="two_non_spiral_aug"):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through each image path
    for idx, image_path in enumerate(image_paths):
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image {image_path}. Skipping.")
            continue
        
        # Randomly select 3 augmentations from the augmentation list
        selected_augmentations = random.sample(augmentation_list, 3)
        
        # Apply each augmentation in the list separately
        for aug_idx, augmentation in enumerate(selected_augmentations):
            augmented_image = augmentation(image=image)["image"]
            # Save the augmented image
            output_path = os.path.join(output_folder, f"Two_image_{idx+1}_aug_{aug_idx+1}.jpg")
            cv2.imwrite(output_path, augmented_image)





# List of paths for generating the Spirals in total 33 files in Spiral folder
# image_paths = [
#     "image_generation/layer_In1.Cucluster2.png",
#     "image_generation/layer_In2.Cucluster1.png",
#     "image_generation/layer_In2.Cucluster2.png"
# ]

# List of paths for generating the Spirals in total 66 files in Non Spiral folder
# image_paths = [
#     "image_generation/layer_In1.Cucluster0.png",
#     "image_generation/layer_In1.Cucluster1.png",
#     "image_generation/layer_In2.Cucluster0.png", 
#     "image_generation/layer_In2.Cucluster3.png", 
#     "image_generation/layer_In2.Cucluster4.png",
#     "image_generation/layer_In2.Cucluster5.png"
# ]


# folder_path = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Spirals'
# folder_path = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/one_spiral_aug'
# folder_path = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/Non_Spirals'

folder_path = '/home/hiddensand/AKSHET_MT23155/ARCHIVES/Akshet/one_non_spiral_aug'

image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]

# Apply all augmentations and save augmented images
augment_and_save_individual(image_paths)
print("All augmentations applied and saved.")