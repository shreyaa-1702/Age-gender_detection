# Age & Gender Detection Using Deep Learning and OpenCV

A computer vision project that detects a person's **age group** and **gender** from images using pre-trained deep learning models and OpenCV's DNN module. The pipeline automatically detects faces in an image and classifies each detected face into predefined age brackets and gender categories.

---

## 📖 Project Overview

Age and gender estimation from facial images is a key problem in computer vision with applications in surveillance, retail analytics, and human-computer interaction. This project leverages **pre-trained deep learning models** integrated with **OpenCV's DNN module** to perform real-time face detection, followed by age group and gender classification on each detected face.

---

## 📂 Repository Structure

```
age-gender-detection/
│
├── models/                          # Pre-trained model files
│   ├── opencv_face_detector.pbtxt   # Face detection model config
│   ├── opencv_face_detector_uint8.pb # Face detection model weights
│   ├── age_deploy.prototxt          # Age prediction model config
│   ├── age_net.caffemodel           # Age prediction model weights
│   ├── gender_deploy.prototxt       # Gender classification model config
│   └── gender_net.caffemodel        # Gender classification model weights
│
├── images/                          # Sample input images for testing
│
├── age_gender_detection.ipynb       # Main Jupyter Notebook
├── README.md                        # Project overview and instructions
└── requirements.txt                 # Python dependencies
```

---

## 🔍 Dataset & Pre-trained Models

The project uses pre-trained models trained on benchmark facial datasets:

### Age Classification
The model classifies faces into the following age groups:

| Age Group | Range |
|-----------|-------|
| Group 1 | 0 – 2 years |
| Group 2 | 4 – 6 years |
| Group 3 | 8 – 12 years |
| Group 4 | 15 – 20 years |
| Group 5 | 25 – 32 years |
| Group 6 | 38 – 43 years |
| Group 7 | 48 – 53 years |
| Group 8 | 60+ years |

### Gender Classification
| Category | Label |
|----------|-------|
| Male | `Male` |
| Female | `Female` |

---

## 🛠️ Technologies Used

| Category | Tools / Libraries |
|----------|-------------------|
| **Language** | Python 3.x |
| **Computer Vision** | OpenCV |
| **Deep Learning** | OpenCV DNN Module |
| **Pre-trained Models** | Caffe-based Age & Gender models, TensorFlow-based Face Detector |

---

## 🚀 Implementation Steps

### 1. Load Pre-trained Models
- Load the **face detection model** (OpenCV DNN — TensorFlow-based)
- Load the **age prediction model** (Caffe-based)
- Load the **gender classification model** (Caffe-based)

### 2. Preprocess the Input Image
- Read the input image using OpenCV
- Create a **blob** from the image using `cv2.dnn.blobFromImage()`
- Normalize pixel values and resize to the required input dimensions

### 3. Detect Faces
- Pass the preprocessed blob through the face detection network
- Apply a **confidence threshold** to filter weak detections
- Extract bounding box coordinates for each detected face

### 4. Predict Age & Gender
- Crop each detected face region from the original image
- Pass the cropped face through the **age prediction network**
- Pass the cropped face through the **gender classification network**
- Extract the predicted class with the highest confidence score

### 5. Display Results
- Draw **bounding boxes** around each detected face
- Overlay the predicted **age group** and **gender label** as text on the image
- Display the annotated output image

---

## 🔧 Key Components

| Component | Description |
|-----------|-------------|
| `cv2.dnn.readNet()` | Loads the pre-trained model from config and weight files |
| `cv2.dnn.blobFromImage()` | Preprocesses image into a blob for DNN input |
| `net.setInput()` | Sets the blob as input to the network |
| `net.forward()` | Runs a forward pass and returns detection/prediction results |
| `cv2.rectangle()` | Draws bounding boxes on detected faces |
| `cv2.putText()` | Overlays age and gender labels on the output image |

---

## 📊 Output

Each detected face in the input image is annotated with:
- A **bounding box** drawn around the face
- A **label** displaying the predicted gender and age group (e.g., `Male, 25-32`)

---

## 🏁 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreyaa-1702/age-gender-detection.git
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Pre-trained Models**

   Download the following model files and place them in the `models/` directory:

   | File | Purpose |
   |------|---------|
   | `opencv_face_detector.pbtxt` | Face detection config |
   | `opencv_face_detector_uint8.pb` | Face detection weights |
   | `age_deploy.prototxt` | Age model config |
   | `age_net.caffemodel` | Age model weights |
   | `gender_deploy.prototxt` | Gender model config |
   | `gender_net.caffemodel` | Gender model weights |

4. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook age_gender_detection.ipynb
   ```

---

## ⚙️ Requirements

```
opencv-python
numpy
```

Install all at once:
```bash
pip install opencv-python numpy
```

---

## 💡 Key Insights

- The model performs best on **front-facing, well-lit** images with clearly visible faces
- Age prediction outputs a **range** rather than an exact value, which is inherent to the pre-trained model's classification approach
- Multiple faces in a single image are all detected and labeled independently
- Confidence thresholding ensures only high-quality face detections are processed for age and gender prediction

---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## 🙋 Contributing

Contributions and suggestions are welcome. Feel free to open an issue or submit a pull request for any improvements or enhancements.
