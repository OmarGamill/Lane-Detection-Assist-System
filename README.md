## Graduation Project: Lane Detection Assist System

### Project Overview
The **Lane Detection Assist System** is an advanced computer vision project aimed at enhancing vehicle safety and driver assistance. This system leverages state-of-the-art hardware and software technologies to detect lane markings on the road, providing real-time feedback and assistance to the driver. The goal is to reduce the risk of accidents by ensuring the vehicle remains within its designated lane.

This repository contains the Lane Detection Assist System, which includes two different approaches for lane detection using OpenCV and Ultrafast Lane Detection models.

### Hardware Components
- **Raspberry Pi 4**: The primary processing unit for running the lane detection algorithms.
- **Camera Module**: Captures real-time video feed of the road ahead.
- **Ultrasonic Sensors**: Assists in detecting obstacles and measuring distance.
- **Arduino**: Interfaces with sensors and controls the alert mechanisms.
- **GPS Module**: Provides real-time location data to enhance the accuracy of lane detection.
- **LCD Display**: Displays lane detection status and warnings to the driver.
- **Power Supply**: Ensures the system has a reliable power source during operation.

### Software Components
- **Programming Languages**: Python
- **Computer Vision Libraries**: OpenCV
- **Machine Learning Frameworks**: TensorFlow, Keras, PyTorch
- **Operating System**: Raspbian (for Raspberry Pi)
- **Other Tools**: Kaggle and Colap Notebook

### Methodology
1. **Video Processing**: The camera module captures continuous video feed, which is primarily processed using OpenCV to detect lane lines. As a secondary method, the Ultrafast Lane Detector is employed for enhanced performance in specific scenarios.
2. **Edge Detection**: Canny edge detection algorithm in OpenCV is applied to identify the edges of lane markings.
3. **Hough Transform**: Hough Line Transform in OpenCV is used to detect and highlight lane lines in the captured frames.
4. **Ultrafast Lane Detection**: In addition to the OpenCV methods, the Ultrafast Lane Detector model is used for robust and real-time lane detection, leveraging deep learning techniques.
5. **Obstacle Detection**: Ultrasonic sensors continuously measure the distance to potential obstacles and trigger alerts if necessary.
6. **Integration and Testing**: The system is integrated with the hardware components and thoroughly tested under various driving conditions.

## Repository Structure

### 1. LaneDetection_openCV

This folder contains the following files:

1. **LaneModule.py**: Main module for lane detection using OpenCV.
2. **WebcamModule.py**: Module for capturing webcam input.
3. **picker_script.py**: Script for selecting parameters.
4. **utils.py**: Utility functions used in lane detection.

**Start point**: `LaneModule.py`

### 2. UlterFastLaneDetection

This folder contains:

- **models**: This sub-folder contains the pre-trained models for Ultrafast Lane Detection.
- **utilities**:
  - `backbone.py`: Backbone network used in the model.
  - `model.py`: Model architecture for lane detection.
  - `ultrafastLaneDetector.py`: Main detector class using Ultrafast Lane Detection.

**Start point**: `videoLaneDetection.py`

### 3. Main Files

1. **main.py**: Script to choose between the two-lane detection systems using command-line arguments.
2. **requirements.txt**: List of dependencies required to run the project.

## How to Use and Run the Repository

### Installation

1. Clone the repository:
   
  ```bash
  git clone https://github.com/your-username/LaneDetectionAssistSystem.git
  cd LaneDetectionAssistSystem

### Conclusion
The **Lane Detection Assist System** is a comprehensive project that demonstrates the practical application of computer vision and machine learning in enhancing vehicle safety. By combining robust hardware with advanced software techniques, this project aims to contribute to the development of intelligent driver assistance systems.

### Contact
For more details about this project, feel free to contact me:
- **Email**: omar.gamil@example.com
- **LinkedIn**: [Omar Gamil LinkedIn](https://www.linkedin.com/in/omar-gamil)
- **GitHub**: [Omar Gamil GitHub](https://github.com/your-username)
