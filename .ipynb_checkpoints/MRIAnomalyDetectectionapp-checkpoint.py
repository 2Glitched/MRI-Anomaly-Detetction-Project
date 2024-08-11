
import pydicom
import os
import pydicom.data
import numpy as np
import cv2
import sklearn

from sklearn.metrics import mean_squared_error


# Function to preprocess the images (resize and normalize)
def preprocess_image(image_array, target_size=(256, 256)):
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image / np.max(resized_image)
    return normalized_image


dicom_directory = r'C:\Users\hamna\Downloads\archive\ST000001\DicomnormalBrain'


def load_and_preprocess_images(dicom_directory):
    if not os.path.exists(dicom_directory):
        print(f"Error: The directory '{dicom_directory}' does not exist.")
        return []

    images = []
    for file in sorted((os.listdir(dicom_directory))):
        if file.endswith('.dcm'):
            file_path = os.path.join(dicom_directory, file)
            dicom_data = pydicom.dcmread(file_path)
            image_array = dicom_data.pixel_array
            preprocessed_image = preprocess_image(image_array)
            images.append(preprocessed_image)
    return images

# Extract features (e.g., pixel mean and standard deviation)


def extract_features(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    return mean, std_dev


# Compare features of test images to normal features
def compare_features(normal_features, test_features):

    normal_mean, normal_std = normal_features
    results = []
    for test_feature in test_features:
        test_mean, test_std = test_feature
        distance = np.sqrt((normal_mean - test_mean)**2 + (normal_std - test_std)**2)
        results.append(distance)
    return results


# Generate a report based on comparison results
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
    print(f"Report content:\n")
    with open(output_file, 'r') as file:
        print(file.read())


# Main function to process normal and test scans
def main(normal_directory, test_directory):
    print(f"Loading normal images from: {normal_directory}")
    normal_images = load_and_preprocess_images(normal_directory)
    print(f"Number of normal images loaded: {len(normal_images)}")
    normal_features = [extract_features(normal_images[0])] if normal_images else [(0, 0)]

    print(f"Loading test images from: {test_directory}")
    test_images = load_and_preprocess_images(test_directory)
    print(f"Number of test images loaded: {len(test_images)}")
    test_features = [extract_features(img) for img in test_images] if test_images else []

    distances = compare_features(normal_features[0], test_features)
    print(f"Number of distances computed: {len(distances)}")

    generate_report(distances)
# Example directories


normal_directory = r'C:\Users\hamna\Downloads\archive\ST000001\DicomnormalBrain'
test_directory = r'C:\Users\hamna\Downloads\MSBraiin Scan\ST000001\SE000006'

main(normal_directory, test_directory)