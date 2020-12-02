from sklearn.model_selection import GridSearchCV
from utils import mnist_reader
import matplotlib as plt
import pickle

# Load test data
x_test, y_test = mnist_reader.load_mnist("data/test", kind="t10k")

# Load models
svc_model = pickle.load(open("models/svc-class.ser", "rb"))
gnb_model = pickle.load(open("models/gnb-class.ser", "rb"))
dtc_model = pickle.load(open("models/dtc-class.ser", "rb"))

# Define scores
svc_score = 0
gnb_score = 0
dtc_score = 0

# Update interval
current_step = 0
update_step = 10

# Iterate through test data
for x, y in zip(x_test, y_test):
    if svc_model.predict(x) == y: svc_score+=1
    if gnb_model.predict(x) == y: gnb_score+=1
    if dtc_model.predict(x) == y: dtc_score+=1
    
    if current_step % update_step == 0:
        svc_acc = (svc_score/current_step) * 100
        gnb_acc = (dtc_score/current_step) * 100
        dtc_acc = (dtc_score/current_step) * 100
        print("TEST " + str(current_step) + ": (SVC: " + str(svc_acc) + "%, GNB: " + str(gnb_acc) + "%, DTC: " + str(dtc_acc) + "%)")
    
    current_step += 1