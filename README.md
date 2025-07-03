High-Performance Vehicle and License Plate Detector using YOLOv8

Objective: To develop a robust, real-time object detection model capable of accurately identifying vehicles and localizing their license plates in a variety of real-world conditions.

Key Technologies: Python, YOLOv8, PyTorch, OpenCV, Roboflow, Google Colab

Methodology:

1. Data Curation & Cleaning: Assembled a custom dataset of vehicle images specific to the Uzbekistan region. Implemented Python scripts to programmatically find and remove duplicate images, ensuring a high-quality baseline dataset.

2. Precise Annotation: Meticulously annotated a dataset of 250+ images with two classes: car and license_plate, using bounding boxes. The annotation process was iteratively refined to ensure high precision, correcting initial errors like the use of polygon masks for a detection task.

3. Data Augmentation & Preprocessing: Leveraged the Roboflow platform to build a powerful data pipeline. Applied strategic preprocessing (Resize, Auto-Orient) and a suite of augmentations (Brightness, Rotation, Exposure, Blur) to create a diverse and robust training set of over several images. This step was critical for model generalization.

4. Model Training & Validation:
    - Utilized transfer learning by fine-tuning a pre-trained YOLOv8s model on the custom dataset.
    - Training was performed on a Google Colab instance with an NVIDIA T4 GPU, reducing training time from an estimated 48 hours to just 6 minutes.
    - Monitored training progress and used Early Stopping to prevent overfitting and select the optimal model.

Results & Performance:

The final model achieved outstanding performance on an unseen test set, demonstrating its accuracy and reliability:

  - mAP50-95: 84.9% (A very high score indicating precise bounding box prediction)
  - mAP50 (Recall): 99.5% (Near-perfect ability to find all objects)
  - License Plate Precision: 99.3%
  - Inference Speed: ~50 FPS on a T4 GPU, making it suitable for real-time video processing.
