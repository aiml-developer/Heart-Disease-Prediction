# ❤️ Heart Disease Prediction App

## 📋 Project Overview
This project is a Machine Learning application designed to predict the likelihood of heart disease in a patient based on medical attributes. It utilizes a **Random Forest Classifier** trained on patient physiological data and provides a user-friendly web interface using **Streamlit**.

## 📊 Dataset Info
The dataset contains patient vitals and health markers. Key features include:

| Feature | Description |
| :--- | :--- |
| **Age** | Age of the patient in years |
| **Sex** | 0 = Female, 1 = Male |
| **Chest Pain Type** | 1: Typical Angina, 2: Atypical Angina, 3: Non-anginal Pain, 4: Asymptomatic |
| **Resting BP** | Resting blood pressure (mm Hg) |
| **Cholesterol** | Serum cholesterol (mg/dl) |
| **Fasting BS** | Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No) |
| **Resting ECG** | 0: Normal, 1: ST-T Wave Abnormality, 2: LV Hypertrophy |
| **Max Heart Rate** | Maximum heart rate achieved |
| **Exercise Angina** | Exercise-induced angina (1 = Yes, 0 = No) |
| **Oldpeak** | ST depression induced by exercise relative to rest |
| **ST Slope** | Slope of the peak exercise ST segment (1: Upward, 2: Flat, 3: Downward) |
| **Target** | **0 = Normal**, **1 = Heart Disease** |

## ⚙️ Methodology
1.  **Data Preprocessing**:
    *   Dataset loaded using Pandas.
    *   Feature scaling applied using `StandardScaler` to normalize numerical values (BP, Cholesterol, Heart Rate, etc.).
    *   Data split into training (80%) and testing (20%) sets.
2.  **Model Training**:
    *   Algorithm: **Random Forest Classifier** (`n_estimators=100`).
    *   Model and Scaler artifacts are saved as `.pkl` files for inference.
3.  **Deployment**:
    *   Interactive web application built with **Streamlit** to accept user inputs and display real-time predictions.

## 🛠 Tech Stack
*   **Python 3.x**
*   **Streamlit** (Frontend Web App)
*   **Scikit-Learn** (Machine Learning)
*   **Pandas & NumPy** (Data Manipulation)
*   **Joblib** (Model Serialization)

## 📦 Installation Steps

1.  **Clone the Repository** (if using Git):
    ```
    git clone https://github.com/aiml-developer/Heart-Disease-Prediction
    cd Heart-Disease-Prediction
    ```

2.  **Create a Virtual Environment**:
    ```
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```
    pip install -r requirements.txt
    ```

## 🚀 Usage

### Step 1: Train the Model
Run the training script to generate the model and scaler files (`heart_disease_model.pkl`, `scaler.pkl`).

python train_model.py


### Step 2: Run the Application
Launch the Streamlit app.

streamlit run app.py


## 📈 Model Training
The `train_model.py` script performs the following:
*   Loads `dataset.csv`.
*   Splits data into `X_train` and `X_test`.
*   Scales features using `StandardScaler`.
*   Fits a `RandomForestClassifier`.
*   Exports the trained model and scaler using `joblib`.

## 🖥 Demo
Once the app is running, you will see a form asking for patient details:
*   Input the **Age**, **Sex**, **Blood Pressure**, etc.
*   Click **"Predict Result"**.
*   The app will display:
    *   ⚠️ **High Risk of Heart Disease** (with probability %)
    *   ✅ **Normal - Low Risk** (with probability %)

<video src="https://github.com/user-attachments/assets/744792c6-0926-452b-b8a9-3b723ae2f938" controls width="700"></video>

## 📝 Conclusion
This project demonstrates an end-to-end Machine Learning workflow, from data processing to model deployment, providing a practical tool for early health risk assessment.
