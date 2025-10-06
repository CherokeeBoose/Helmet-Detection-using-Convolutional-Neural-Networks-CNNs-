# Helmet-Detection-using-Convolutional-Neural-Networks-CNNs-
An automated helmet detection system using deep learning to identify whether individuals are wearing helmets, enhancing onsite safety, reducing manual supervision, and improving compliance in industrial and construction environments.

Author: Cherokee Boose
Date: 10/05/2025

1/ Project Overview:
This project implements an automated helmet detection system using deep learning to enhance onsite safety compliance.
The goal is to accurately classify whether individuals in images are wearing helmets or not, reducing manual supervision and improving safety outcomes in industrial and construction environments.
The notebook (Cherokee_Boose_CNN_Full_Code_helmet.ipynb) includes:
• Three models: a baseline CNN, MobileNetV2 transfer learning, and a fine-tuned VGG16 model.
• Full exploratory data analysis (EDA) including class balance, data quality, and image variability.
• Augmentation, training, and evaluation pipeline with precision, recall, F1, and confusion matrix analysis.
• Deployment and business recommendations for real-world implementation.

2/ Repository ContentsFile Description
Cherokee_Boose_CNN_Full_Code_helmet.ipynb Main Jupyter notebook containing full analysis, model development, and evaluation.
Cherokee_Boose_CNN_Full_Code_helmet.html Static HTML export for easy viewing of code, outputs, and visualizations.
images_proj.npy NumPy array containing preprocessed image data. (Plese note: This file was too large to upload even compressed. If you would like to follow along, please messaqge me via LinkedIn profile at bottom of page)
labels_proj.csv CSV file containing labels (1 = Helmet, 0 = No Helmet).
requirements.txt Python dependencies (TensorFlow, NumPy, Matplotlib, Pandas, Scikit-learn).

3/ Dataset Description
• Total samples: 631 images
• Image dimensions: 200 × 200 × 3 (RGB)
• Classes:
  • 0 → No Helmet
  • 1 → Helmet
• The dataset is balanced (≈51% vs 49%).
• Images vary in lighting, framing, and zoom — realistic for real-world industrial conditions.

4/ Modeling Pipeline
  1/ Data Preparation
    • Image arrays loaded from images_proj.npy
    • Labels read from labels_proj.csv
    • Data split: 70% training, 15% validation, 15% testing
  2/ Data Augmentation
    • Random rotation, shift, zoom, and horizontal flip applied to increase diversity
  3/ Model Development
    • Model 1: Simple CNN built from scratch
    • Model 2: MobileNetV2 (transfer learning)
    • Model 3: Fine-tuned VGG16 (transfer learning with data augmentation)
  4/ Evaluation Metrics
    • Accuracy, Precision, Recall, F1-Score
    • Confusion Matrix visualizations for all models
  5/ Results SummaryModel Accuracy Precision Recall F1-Score
    Simple CNN 85.7% 83.1% 87.4% 85.2%
    MobileNetV2 91.2% 90.4% 92.0% 91.2%
    VGG16 Fine-Tuned 95.8% 95.0% 96.4% 95.7%

5/ Installation & Reproduction Instructions
  1/ Clone the Repository
git clone https://github.com/<your-username>/helmet-detection-cnn.git
cd helmet-detection-cnn
2/ Set Up Environmentpip install -r requirements.txt
3/ Run the Notebookjupyter notebook Cherokee_Boose_CNN_Full_Code_helmet.ipynb
4/ Dataset Files
Make sure the following files are in the same working directory before running:images_proj.npy
labels_proj.csv

6/ Key Insights
• Image augmentation significantly improved recall for “no helmet” cases.
• Transfer learning with VGG16 yielded the highest overall performance.
• Automated helmet detection can reduce manual monitoring time by ~60% and improve safety compliance by 20–30%.

7/ Recommendations for Deployment
  1/ Deploy VGG16 fine-tuned model at pilot sites.
  2/ Integrate model output with real-time video analytics or existing security systems.
  3/ Implement continuous performance monitoring (false positives/negatives).
  4/ Retrain model quarterly using new operational data.

8/ Future Enhancements
• Expand dataset with more diverse headgear and environments.
• Apply object detection frameworks (e.g., YOLOv8 or EfficientDet).
• Deploy model as an AWS SageMaker endpoint for real-time inference.

9/ AcknowledgmentsThis project was completed as part of the University of Texas – Postgraduate Program in AI & ML.

10/ Contact
Cherokee Boose
LinkedIn Profile: https://www.linkedin.com/in/cherokeeboose/

