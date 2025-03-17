# Sentiment Analysis using Logistic Regression

## Overview
This project implements **sentiment analysis** on customer reviews using **Logistic Regression**. The dataset is extracted from compressed `.bz2` files, preprocessed, and then vectorized using **CountVectorizer**. Finally, a logistic regression model is trained to classify the sentiment as either **positive** or **negative**.

---

## Features
- **Data Extraction**: Reads and extracts labeled data from `train.ft.txt.bz2` and `test.ft.txt.bz2`.
- **Text Preprocessing**: Removes punctuation, converts to lowercase, and filters non-ASCII characters.
- **Vectorization**: Uses `CountVectorizer` to convert text into numerical features.
- **Model Training**: Trains a logistic regression classifier with different values of `C` (regularization strength).
- **Evaluation**: Computes accuracy scores on validation data.

---

## Dependencies
To run this project, you need the following libraries:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```

---

## Dataset
The dataset consists of compressed `.bz2` files containing labeled text reviews.
- **train.ft.txt.bz2** - Training dataset
- **test.ft.txt.bz2** - Test dataset

Each line in the dataset consists of:
- **Label** (`1` or `0`): Represents positive or negative sentiment.
- **Review Text**: Customer review content.

---

## Code Implementation
### **1. Data Extraction**
```python
import bz2
import numpy as np

def get_labels_and_texts(file):
    labels = []
    texts = []
    for line in bz2.BZ2File(file):
        x = line.decode("utf-8")
        labels.append(int(x[9]) - 1)  # Convert 1-based to 0-based indexing
        texts.append(x[10:].strip())  # Extract review text
    return np.array(labels), texts

train_labels, train_texts = get_labels_and_texts('train.ft.txt.bz2')
test_labels, test_texts = get_labels_and_texts('test.ft.txt.bz2')
```

### **2. Text Preprocessing**
```python
import re

NON_ALPHANUM = re.compile(r'[\W]')
NON_ASCII = re.compile(r'[^a-z0-1\s]')

def normalize_texts(texts):
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts

train_texts = normalize_texts(train_texts)
test_texts = normalize_texts(test_texts)
```

### **3. Vectorization using CountVectorizer**
```python
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(train_texts)
X = cv.transform(train_texts)
X_test = cv.transform(test_texts)
```

### **4. Model Training & Evaluation**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, train_labels, train_size=0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print(f"Accuracy for C={c}: {accuracy_score(y_val, lr.predict(X_val))}")
```

**Output Accuracy:**
```
Accuracy for C=0.01: 0.744
Accuracy for C=0.05: 0.744
Accuracy for C=0.25: 0.752
Accuracy for C=0.5: 0.752
Accuracy for C=1: 0.76
```

### **5. Prediction Example**
```python
print(lr.predict(X_test[29]))  # Output: array([0])
print(test_labels[29])  # Output: 0
print(test_texts[29])  # Sample review text
```

---

## Results & Conclusion
- The logistic regression model achieved an **accuracy of 76%**.
- Increasing `C` improved accuracy but with diminishing returns.
- Preprocessing plays a key role in improving performance.

---

## Future Enhancements
🔹 **Use TF-IDF Vectorization** instead of CountVectorizer for better feature weighting.  
🔹 **Experiment with Deep Learning Models** (LSTMs, BERT) for improved accuracy.  
🔹 **Hyperparameter Tuning** using Grid Search or Random Search.  


---

## Author
👨‍💻 **Your Name**  


