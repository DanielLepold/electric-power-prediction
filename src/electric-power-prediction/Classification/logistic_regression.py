import logging
import numpy as np
import seaborn as sns
import os
import matplotlib as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import utils
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc



def create_heatmap(cf_matrix, folder_path):
  # Create heatmap
  sns.heatmap(cf_matrix, annot=True, cbar=False)

  # Save the plot
  plt.savefig(os.path.join(folder_path, "heatmap.png"))

  # Show the plot
  plt.show()

def perform_regression(X_train,X_test, y_train,y_test, folder):
  folder_path = folder + "/Logistic"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
  logging.info("\n-------------------------------------------------------------"
               "--------------------------------")
  logging.info("Logistic regression started.")

  # convert y values to categorical values
  lab = preprocessing.LabelEncoder()
  y_transformed = lab.fit_transform(y_train)

  clf = LogisticRegression(multi_class='auto')
  clf.fit(X_train, y_transformed)

  # Logistic prediction
  y_p_train = clf.predict(X_train)
  y_p_test = clf.predict(X_test)
  #y_p_test = clf.predict(X_test_clf)

  #np.unique(y_transformed)
  #cf_matrix = confusion_matrix(y_transformed, y_p_train)

  # This takes soo much time I could not run it
  #create_heatmap(cf_matrix,folder_path)
  logging.info(f"{clf.coef_}")

  logging.info(f"Accuracy score, train: \n{accuracy_score(y_train, y_p_train)}")
  logging.info(f"Accuracy score, test: \n{accuracy_score(y_test, y_p_test)}")

  logging.info(f"Balanced accuracy score, train: \n{balanced_accuracy_score(y_train, y_p_train)}")
  logging.info(f"Balanced accuracy score, test: \n{balanced_accuracy_score(y_test, y_p_test)}")

  logging.info(f"F1 score, train: \n{f1_score(y_train, y_p_train,average=None)}")
  logging.info(f"F1 score, test: \n{f1_score(y_test, y_p_test,average=None)}")

  logging.info(f"Precision score, train: \n{precision_score(y_train, y_p_train,average=None)}")
  logging.info(f"Precision score, test: \n{precision_score(y_test, y_p_test,average=None)}")

  logging.info(f"Recall score, train: \n{recall_score(y_train, y_p_train,average=None)}")
  logging.info(f"Recall score, test: \n{recall_score(y_test, y_p_test,average=None)}")




  logging.info("Logistic regression finished.")
