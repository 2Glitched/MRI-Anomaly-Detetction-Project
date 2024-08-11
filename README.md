# MRI Anomaly Detection Project

## Description

This project focuses on detecting anomalies in MRI brain scans using machine learning techniques. The provided code processes DICOM images, extracts features, and compares them to identify potential anomalies. This repository includes sample DICOM images for demonstration purposes.

## Features
* Preprocessing of DICOM images: resizing and normalization
* Feature extraction: calculates mean and standard deviation
* Anomaly detection: compares features and generates a report
* Sample DICOM images for normal brain MRI included
  
## Installation
Clone the Repository:
```bash
git clone https://github.com/2Glitched/mri-anomaly-detection.git
cd mri-anomaly-detection
```

### Install Dependencies
```
pip install -r requirements.txt
```
### Usage

Set Up Environment Variables

Create a .env file in the root directory of the project with the following content:
```
DICOM_DIRECTORY=path_to_normal_images
TEST_DIRECTORY=path_to_test_images
```
Replace path_to_normal_images and path_to_test_images with the paths to your directories.

## Run the Script

Execute the script to process the images and generate the anomaly detection report:

```
python your_script_name.py
```
The script will load images from the specified directories, process them, and generate a report.txt file with the results.

## Example Python Code
Here is an example of the preprocess_image function used in the project:
```
python

import cv2
import numpy as np

def preprocess_image(image_array, target_size=(256, 256)):
    resized_image = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image / np.max(resized_image)
    return normalized_image
```

## Data
- Sample Data: The repository includes sample DICOM images located in the sample_data/brain_mri/ directory. These images are provided for demonstration purposes.

- Usage: You can replace these sample images with your own DICOM files for more comprehensive testing.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

1. Fork the repository.
2. Create a new branch: git checkout -b feature/YourFeature.
3. Make your changes and commit them: git commit -am 'Add new feature'.
4. Push to the branch: git push origin feature/YourFeature.
5. Create a new Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
Libraries: This project uses several Python libraries, including pydicom, numpy, opencv-python, and scikit-learn.
Sample Data: The sample DICOM images are sourced from keggel.



Acknowledgements
Libraries: This project uses several Python libraries, including pydicom, numpy, opencv-python, and scikit-learn.
Sample Data: The sample DICOM images are sourced from [source, if applicable].
