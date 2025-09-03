# Psoriasis Detection ( please read all the isntructions otherwise it will not work correctly)

Psoriasis Skin Disease Detection using ML & Image Processing

Developed a machine learning model to classify skin images as Normal or Psoriasis using image preprocessing and an SVM model.

Created an interactive Streamlit web application enabling users to upload images and receive instant predictions.

Technologies: Python, OpenCV, Scikit-learn, Streamlit, NumPy, Joblib

# Dataset

The dataset size is large, so please download it from the link provided.


After downloading, place the dataset folder inside the project folder.

# Model Training

Once all files/folders are in place, open the terminal and run:

python model_train.py (This will create a .pkl file named psoriasis_model.pkl.)

# Running the Project
To run the project, open the terminal and run:
streamlit run app.py (This will open a web page where you can upload an image and detect Psoriasis.)

# Folder Structure
Psoriasis-detection/           (/ indicates folder)
│
├─ dataset/
│   ├─ normal/
│   └─ psoriasis/
│
├─ app.py
├─ model_train.py
└─ psoriasis_model.pkl

# plss download all the required libraries given in "requirement.text file " (plss see the file section)







