## Graduation Project: Lane Detection Assist System

![Lane Detection Example](https://github.com/OmarGamill/Lane-Detection-Assist-System/blob/main/car_fully_system.gif)
![Demo](https://github.com/OmarGamill/Lane-Detection-Assist-System/blob/main/Demo.gif)

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
<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/OmarGamill/Lane-Detection-Assist-System/blob/main/images/car_L0.jpg" alt="Lane Detection Example" width="300"/>
  <img src="https://github.com/OmarGamill/Lane-Detection-Assist-System/blob/main/images/car_L1.jpg" alt="Lane Detection Result" width="300"/>
</div>

<div style="display: flex; justify-content: space-around;">
  <img src="https://github.com/OmarGamill/Lane-Detection-Assist-System/blob/main/images/car_L2.jpg" alt="Lane Detection Example" width="300"/>
  <img src="https://github.com/OmarGamill/Lane-Detection-Assist-System/blob/main/images/car_L3.jpg" alt="Lane Detection Result" width="300"/>
</div>


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

1. `LaneModule.py`: Main module for lane detection using OpenCV.
2. `WebcamModule.py`: Module for capturing webcam input.
3. `picker_script.py`: Script for selecting parameters.
4. `utils.py`: Utility functions used in lane detection.

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

1. `main.py`: Script to choose between the two-lane detection systems using command-line arguments.
2. `requirements.txt`: List of dependencies required to run the project.

### Dataset

This project uses the Tusample dataset. You can download it from [Kaggle](https://www.kaggle.com/datasets/manideep1108/tusimple).

### Training the Model

To train the model using Google Colab, follow these steps:

1. Open [Google Colab](https://colab.research.google.com/drive/1FACfF5yCfwlddUy5ZHfInZ4Wqf5xgkkF?usp=sharing).
2. Run the training script.

### Testing the Model

You can test the model with a video input to see the lane detection in action. Below is an example command to run the test:

    ```bash
    python videoLaneDetection.py --video path/to/video.mp4

### Results

Watch the result of lane detection in action:

- **Demo Video:** [Link to demo video](https://drive.google.com/file/d/1nqunV3IaRufQ2DzsFlCOwpjJOagzz_Gl/view?usp=sharing)
- **Result Video:** [Link to result video](https://drive.google.com/file/d/192FndntNAXjbCHP1dH7iQlgtS830WMG7/view?usp=sharing)

## How to Use and Run the Repository

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OmarGamill/Lane-Detection-Assist-System.git

2. Navigate to the project directory:
    ```bash
    cd Lane-Detection-Assist-System
    
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the main script:
   ```bash
   python main.py [file] [--video path/to/video.mp4]
   
5. To run the OpenCV-based lane detection:
   ```bash
   python main.py LaneModule --video path/to/video.mp4
   
6. To run the Ultrafast Lane Detection:
   ```bash
   python main.py videoLaneDetection --video path/to/video.mp4
   

### Conclusion
The **Lane Detection Assist System** is a comprehensive project that demonstrates the practical application of computer vision and machine learning in enhancing vehicle safety. By combining robust hardware with advanced software techniques, this project aims to contribute to the development of intelligent driver assistance systems.

## Contributing

Contributions are welcome! Please open issues or submit pull requests to improve the project.

## Contact
- **Email**: omargamel258@gmail.com
- **LinkedIn**: [Omar Gamil LinkedIn](https://www.linkedin.com/in/omar-gamel-8628531b3/)
- **GitHub**: [Omar Gamil GitHub](https://github.com/OmarGamill)
