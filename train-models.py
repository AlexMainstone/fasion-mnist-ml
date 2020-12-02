from sklearn.model_selection import GridSearchCV
from sklearn import svm, naive_bayes, tree
from utils import mnist_reader
import pickle
import os

import matplotlib.pyplot as plt

def train_model(x, y, classifier, parameters):
    # Define Hyperparameter Estimator
    gridsearch = GridSearchCV(classifier, parameters)

    # Train
    print("Training...")
    gridsearch.fit(x, y)
    print("Trained!")
    
    return gridsearch

# Load our training data
x_train, y_train = mnist_reader.load_mnist("data/train", kind="train")

# Make a models folder if it doesn't exist
if not os.path.exists("models"):
    os.mkdir("models")

# Train SVC
print("SVC:")
svc_param = {}
svc_model = train_model(x_train, y_train, svm.SVC(), svc_param)
pickle.dump(svc_model, open("models/svc-class.ser", "wb"))

# Train Naive Bayes
print("GNB:")
gnb_param = {}
gnb_model = train_model(x_train, y_train, naive_bayes.GaussianNB(), gnb_param)
pickle.dump(gnb_model, open("models/gnb-class.ser", "wb"))

# Train Decision Tree
print("DTC:")
dtc_param = {}
dtc_model = train_model(x_train, y_train, tree.DecisionTreeClassifier(max_depth=5), dtc_param)
pickle.dump(dtc_model, open("models/dtc-class.ser", "wb"))

print("Trained all models. Run test-models.py to test.")