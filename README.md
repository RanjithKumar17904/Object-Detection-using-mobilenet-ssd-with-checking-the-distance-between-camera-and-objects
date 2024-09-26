# Object Detection Using MobileNet SSD with Distance Estimation

This project demonstrates object detection using the **MobileNet SSD** model and includes functionality to estimate the distance between the camera and detected objects. The primary goal is to detect objects in a scene and calculate their relative distances from the camera in real-time.

## Features

- Real-time object detection using MobileNet SSD.
- Distance estimation between the camera and detected objects.
- Easy-to-extend code for adding more features.
  
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Object-Detection-using-mobilenet-ssd-with-checking-the-distance-between-camera-and-objects.git
   cd Object-Detection-using-mobilenet-ssd-with-checking-the-distance-between-camera-and-objects
   ```

2. **Install the required dependencies:**
   Make sure you have Python 3.x installed. You can then install the necessary packages via `pip`:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Running Object Detection:**
   After installing the dependencies, run the following command to start object detection:
   ```bash
   python object_detection.py
   ```

2. **Distance Estimation:**
   The system calculates the distance from the camera to the detected object using a pre-calibrated focal length. You can adjust the parameters based on your camera setup inside the code.

3. **Configuration:**
   The code can be modified to change parameters like detection threshold, distance formula, or camera calibration by editing the configuration inside `config.py`.

## Dependencies

- OpenCV
- TensorFlow
- NumPy
- MobileNet-SSD model (pre-trained)

You can install these dependencies via the `requirements.txt` file.

## Results

Object detection output will show bounding boxes for each detected object in the frame. The distance from the camera is printed on the bounding box. The performance will vary depending on the quality of the camera used.

## Contributing

Feel free to submit issues or pull requests if you want to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
