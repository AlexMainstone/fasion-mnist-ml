from sklearn import svm
from sklearn.model_selection import GridSearchCV

from utils import mnist_reader
import pickle

import matplotlib.pyplot as plt

# Load fasion dataset
x_train, y_train = mnist_reader.load_mnist("data/train", kind="train")
x_test, y_test = mnist_reader.load_mnist("data/test", kind="t10k")

# Define Classified
classifier = svm.SVC(verbose=1)

# Define Hyperparameter Estimator
parameters = {
    'kernel':['rbf'],
    'C':[1],
    'gamma':["scale"]
}
gridsearch = GridSearchCV(classifier, parameters)

# Train SVC
print("Training SVC...")
gridsearch.fit(x_train, y_train)
print("Trained!")

print(sorted(gridsearch.cv_results_.keys()))

# Test SVC
print("Testing SVC...")
prediction = gridsearch.predict(x_test)
print("Tested!")

# Calculate correct predictions
correct = 0
for p, a in zip(prediction, y_test):
    if p == a:
        correct += 1
print("SVC correctly identified: " + str(correct) + "/" + str(len(prediction)))

# Serialize Model
ans = input("save model? (y/n) :>")
if ans == "y":
    pickle.dump(gridsearch, open("models/svc-class.ser", "wb"))