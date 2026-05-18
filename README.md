# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
**Step 1:** Import the spam dataset, clean the data, and convert text messages into numerical format using TF-IDF vectorization.

**Step 2:** Divide the dataset into training data and testing data for model evaluation.

**Step 3:** Train the Support Vector Machine (SVM) classifier using the training dataset to identify spam and ham messages.

**Step 4:** Test the trained model with unseen data, predict the message category, and evaluate the performance using a confusion matrix visualization.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by:SARFARAZ I 
RegisterNumber: 2122225230252
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("spam (1).csv", encoding='latin-1')
data = data[['v1','v2']]
data.columns = ['label','message']
data['label'] = data['label'].map({'ham':0, 'spam':1})
X = data['message']
y = data['label']
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show() 
*/
```

## Output:

<img width="482" height="400" alt="image" src="https://github.com/user-attachments/assets/a23a4f37-3406-405b-9909-d91486f32b3f" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
