import pydicom
import os
import numpy as np
import cv2
from sklearn.metrics import mean_squared_error

# Preprocess the image (resize and normalize)
def preprocess_image(image_array, target_size=(256, 256)):
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image / np.max(resized_image)
    return normalized_image

# Set paths (use raw strings to avoid backslash errors)
dicom_directory = r'C:\Users\morning\Desktop\DicomnormalBrain'
test_directory = r'C:\Users\morning\Desktop\MSBrain Scan\ST000001\SE000007'

# Load and preprocess DICOM images
def load_and_preprocess_images(directory):
    if not os.path.exists(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return []

    images = []
    for file in sorted(os.listdir(directory)):
        if file.endswith('.dcm'):
            file_path = os.path.join(directory, file)
            try:
                dicom_data = pydicom.dcmread(file_path)
                image_array = dicom_data.pixel_array
                preprocessed_image = preprocess_image(image_array)
                images.append(preprocessed_image)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    return images

# Extract features from image
def extract_features(image):
    return np.mean(image), np.std(image)

# Compare test features to normal reference
def compare_features(normal_features, test_features):
    normal_mean, normal_std = normal_features
    results = []
    for test_mean, test_std in test_features:
        distance = np.sqrt((normal_mean - test_mean)**2 + (normal_std - test_std)**2)
        results.append(distance)
    return results

# Generate a report
def generate_report(distances, output_file='report.txt', threshold=0.1):
    with open(output_file, 'w') as file:
        file.write("Comparison Report\n")
        file.write("=================\n")
        if not distances:
            file.write("No test images were processed or compared.\n")
        else:
            for i, distance in enumerate(distances):
                status = "Anomaly Detected" if distance > threshold else "No Anomaly"
                file.write(f"Test Image {i+1}: Distance = {distance:.4f}, Status: {status}\n")
    print(f"Report generated: {output_file}")
    with open(output_file, 'r') as file:
        print(file.read())

# Main script
def main():
    print(f"Loading normal images from: {dicom_directory}")
    normal_images = load_and_preprocess_images(dicom_directory)
    print(f"Number of normal images loaded: {len(normal_images)}")

    if not normal_images:
        print("No normal images found. Exiting.")
        return

    normal_features = extract_features(normal_images[0])  # Using first normal as reference

    print(f"Loading test images from: {test_directory}")
    test_images = load_and_preprocess_images(test_directory)
    print(f"Number of test images loaded: {len(test_images)}")

    test_features = [extract_features(img) for img in test_images] if test_images else []
    distances = compare_features(normal_features, test_features)

    print(f"Number of distances computed: {len(distances)}")
    generate_report(distances)


if __name__ == "__main__":
    main()
