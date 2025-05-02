import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay

X, Y = make_moons(n_samples=1000, noise=0.2, random_state=1)


XTR, XTS, YTR, YTS = train_test_split(X, Y,
                                      test_size=0.2,  # percentage preserved as test data
                                      random_state=1, # seed for replication
                                      stratify = Y)   # Preserves distribution of y

dfTR = pd.DataFrame(XTR, columns=["X" + str(i + 1) for i in range(XTR.shape[1])])
inputs = dfTR.columns
dfTR["Y"] = YTR
output = "Y"

dfTS = pd.DataFrame(XTS, columns=["X" + str(i + 1) for i in range(XTS.shape[1])])
inputs = dfTS.columns
dfTS["Y"] = YTS
output = "Y"

k_values = np.ceil(np.linspace(3, XTR.shape[0] / 2, num=15)).astype("int").tolist()

hyp_grid = {'knn__n_neighbors': k_values} 

num_folds = 10

knn_pipe = Pipeline(steps=[('scaler', StandardScaler()), 
                           ('knn', KNeighborsClassifier())])

knn_gridCV = GridSearchCV(estimator=knn_pipe, param_grid=hyp_grid, 
                        cv=num_folds,
                        return_train_score=True)

knn_gridCV.fit(XTR, YTR) 

dfTR_eval = dfTR.copy()
dfTR_eval['Y'] = YTR 
dfTR_eval['Y_knn_prob_neg'] = knn_gridCV.predict_proba(XTR)[:, 0]
dfTR_eval['Y_knn_prob_pos'] = knn_gridCV.predict_proba(XTR)[:, 1]
dfTR_eval['Y_knn_pred'] = knn_gridCV.predict(XTR)

# Test predicitions dataset
dfTS_eval = dfTS.copy()
dfTS_eval['Y'] = YTS
dfTS_eval['Y_knn_prob_neg'] = knn_gridCV.predict_proba(XTS)[:, 0]
dfTS_eval['Y_knn_prob_pos'] = knn_gridCV.predict_proba(XTS)[:, 1]
dfTS_eval['Y_knn_pred'] = knn_gridCV.predict(XTS)


model = knn_gridCV
fig = plt.figure(constrained_layout=True, figsize=(6, 2))
spec = fig.add_gridspec(1, 3)
ax1 = fig.add_subplot(spec[0, 0]);ax1.set_title('Training'); ax1.grid(False)
ax2 = fig.add_subplot(spec[0, 2]);ax2.set_title('Test'); ax2.grid(False)
ConfusionMatrixDisplay.from_estimator(model, XTR, YTR, cmap="Greens", colorbar=False, ax=ax1, labels=[1, 0])
ConfusionMatrixDisplay.from_estimator(model, XTS, YTS, cmap="Greens", colorbar=False, ax=ax2, labels=[1, 0])
plt.suptitle("Confusion Matrices")
plt.show()
