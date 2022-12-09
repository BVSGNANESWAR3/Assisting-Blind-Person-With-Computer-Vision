# Assisting-Blind-Person-With-Computer-Vision
Mini-Project Semester 7 by: K.Rithvik, MahidharAlam, B.V.S.GNANESWAR Under Guidance of Dr. P.K. Srijith and Athul

# Repository Structure

- main
  - Results- In this folder you can find the metrics of the model
  - code
    - Train_YOLOv5.ipynb - For training the model
    - Detect_YOLOv5.ipynb - Actual code for testing on new videos
    - yolov5
      - detect.py - It contains object detection code and call the functions for action prediction and edge detection
      - Action_Predictor.py - It contains the functions for movement assistance and action prediction
      - Lane2.py - It contains code for edge detection
      - TrainedModels - Contains Pre-trained models (The at present best model is best500.pt)   



## DATASET CREATION
- Tool Used : Image Annotation Lab (IAL) (https://ial.4smartmachines.com/)
- IAL can be used on both GPU and CPU.
- Videos were shot inside the campus and used IAL is used to convert them to frames and label the frames.
- Label a single frame of a video, the IAL tool labels the all other frames of the video using AI.
- Around 60 videos of different cases such that they include various labels were shot each around 20 seconds.
- All the parameters's values in IAL were set to default values.
- FINAL DATASET : (paste the link here)


