# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image

def load_image_with_pillow(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            return np.array(img)
    except Exception as e:
        print(f"Failed to load image with Pillow: {image_path}, {e}")
        return None

def save_image_with_pillow(image_array, output_path):
    try:
        img = Image.fromarray(image_array)
        img.save(output_path)
        print(f'Saved transformed image: {output_path}')
    except Exception as e:
        print(f"Failed to save image with Pillow: {output_path}, {e}")

def remove_transparent_border(img):
    if img.shape[2] == 4:
        alpha_channel = img[:, :, 3]
        non_transparent_mask = alpha_channel != 0
        non_transparent_indices = np.where(non_transparent_mask)
        if len(non_transparent_indices[0]) > 0 and len(non_transparent_indices[1]) > 0:
            y_min, y_max = np.min(non_transparent_indices[0]), np.max(non_transparent_indices[0])
            x_min, x_max = np.min(non_transparent_indices[1]), np.max(non_transparent_indices[1])
            img = img[y_min:y_max+1, x_min:x_max+1, :]
    return img

def ensure_aspect_ratio(img, target_width):
    img_height, img_width = img.shape[:2]
    target_height = int(target_width * 2 / 3)  # 3:2 ratio
    if img_height < target_height:
        top_padding = (target_height - img_height) // 2
        bottom_padding = target_height - img_height - top_padding
        padded_img = cv2.copyMakeBorder(img, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
        return padded_img
    elif img_height > target_height:
        new_width = img_width
        new_height = target_height
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_img
    return img

def apply_matrix3d_transform(image_path, output_path):
    try:
        img = load_image_with_pillow(image_path)
        if img is None:
            raise FileNotFoundError(f'Failed to load image: {image_path}')

        new_width = 450
        new_height = 300
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # CSS matrix3d to OpenCV perspective matrix transformation
        css_matrix3d = np.array([
            [0.0751673, -0.047713, -0.0001107, 51],
            [-0.013814, 0.13081, -0.0002019, 66],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Convert 4x4 matrix to 3x3 matrix for perspective transformation
        matrix3x3 = np.array([
            [css_matrix3d[0][0], css_matrix3d[0][1], css_matrix3d[0][3]],
            [css_matrix3d[1][0], css_matrix3d[1][1], css_matrix3d[1][3]],
            [css_matrix3d[3][0], css_matrix3d[3][1], css_matrix3d[3][3]]
        ])

        # Apply perspective transformation
        transformed_img = cv2.warpPerspective(img_resized, matrix3x3, (img_resized.shape[1], img_resized.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGRA2GRAY)
        _, alpha = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(alpha)
        cropped_img = transformed_img[y:y+h, x:x+w]

        # Create rotation matrix
        center = (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 20, 1.0)  # Rotate left 20 degrees

        # Calculate new canvas size
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_w = int(cropped_img.shape[0] * abs_sin + cropped_img.shape[1] * abs_cos)
        new_h = int(cropped_img.shape[0] * abs_cos + cropped_img.shape[1] * abs_sin)

        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Apply rotation
        rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Remove unnecessary transparent parts after transformation
        rotated_img = remove_transparent_border(rotated_img)

        # Ensure 3:2 aspect ratio with padding
        final_img = ensure_aspect_ratio(rotated_img, new_w)

        save_image_with_pillow(final_img, output_path)

    except Exception as e:
        print(f"Error processing file {image_path}: {e}")

def main():
    default_directory = os.getcwd()
    while True:
        input_folder = input(f"Please specify the input folder (or, Enter key [{default_directory}]): ").strip()
        if not input_folder:
            input_folder = default_directory

        if not os.path.exists(input_folder):
            print("The specified input folder does not exist. Please try again.")
        else:
            break

    supported_formats = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    files_to_process = [file_name for file_name in os.listdir(input_folder) if file_name.lower().endswith(supported_formats)]
    
    if not files_to_process:
        print("No supported image files found in the specified input folder.")
    else:
        output_folder = os.path.join(input_folder, "cubes")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file_name in files_to_process:
            try:
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
                apply_matrix3d_transform(input_path, output_path)
                print(f"Transformed {file_name}.")
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    print("")
    print("Press any key to exit . . .")
    input()

if __name__ == "__main__":
    main()
