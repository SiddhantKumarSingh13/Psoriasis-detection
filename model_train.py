import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib


data = []
labels = []

categories = ['psoriasis', 'normal']

for category in categories:
    path = os.path.join('dataset', category)
    label = categories.index(category)

    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).flatten() / 255.0
            data.append(gray)
            labels.append(label)
        except:
            pass

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

joblib.dump(model, 'psoriasis_model.pkl')
print("Model trained and saved successfully!")
