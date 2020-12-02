from sklearn.model_selection import GridSearchCV
import pickle
import os.path

# load model
gs = pickle.load(open(os.path.dirname(__file__) + "/../models/svc-class.ser", "rb"))

# print some data
print(sorted(gs.cv_results_.keys()))