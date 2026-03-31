#  AI-Powered Healthcare Diagnosis with Generative AI

## Overview
This project is an **AI-powered healthcare diagnosis platform** that combines **traditional machine learning models** with **Generative AI** to assist in disease prediction, prescription analysis, and personalized posture correction.

The system is designed to:
- Predict diseases with high accuracy from patient health data.
- Analyze prescriptions automatically using OCR.
- Generate personalized recovery guidance using GenAI.

##  Features
- **Disease Prediction**  
  Achieves **92% accuracy** using Logistic Regression on a dataset of 500+ patient records.
- **Prescription Analysis**  
  Automated OCR pipeline (powered by **Tesseract**) that processes prescriptions **75% faster** than manual entry.
- **Posture Correction Guidance**  
  Real-time posture detection using **MediaPipe** with **GenAI-generated** recovery tips tailored to each patient.

## 🛠Tech Stack
- **Programming Language:** Python  
- **Machine Learning:** scikit-learn, Logistic Regression, Pandas, NumPy  
- **OCR:** Tesseract  
- **Pose Estimation:** MediaPipe  
- **Generative AI:** OpenAI API / Gemini (for personalized suggestions)  
- **Visualization:** Matplotlib, Seaborn  
- **Environment:** Jupyter Notebook

##  Project Structure
```
ML-model-for-healthcare-using-gen-ai/
│── data/                  # Dataset and sample prescription images
│── notebooks/             # Jupyter notebooks for experiments
│── src/                   # Core source code
│   ├── disease_predictor.py
│   ├── ocr_prescription.py
│   ├── posture_correction.py
│── models/                # Trained model files
│── requirements.txt       # Python dependencies
│── README.md               # Project documentation
```

##  Workflow
1. **Data Preprocessing** → Clean & prepare patient data.
2. **Model Training** → Logistic Regression for disease prediction.
3. **OCR Pipeline** → Extract text from prescriptions.
4. **Posture Detection** → Track key body landmarks using MediaPipe.
5. **GenAI Guidance** → Generate recovery plans using natural language generation.

##  Results
- **Disease Prediction:** 92% accuracy  
- **OCR Processing:** 75% faster workflow  
- **Posture Detection:** Real-time feedback (<100ms latency)

##  How to Run
```bash
# Clone the repository
git clone https://github.com/garv1garv/ML-model-for-healthcare-using-gen-ai.git
cd ML-model-for-healthcare-using-gen-ai

# Install dependencies
pip install -r requirements.txt

# Run the disease prediction model
python src/disease_predictor.py
```

##  License
This project is licensed under the MIT License.
