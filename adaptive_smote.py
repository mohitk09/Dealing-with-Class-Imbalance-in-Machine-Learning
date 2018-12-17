import pandas as pd  # to import csv and for data manipulation
import matplotlib.pyplot as plt  # to plot graph
import seaborn as sns  # for interactve graphs
from adasyn import ADASYN
from sklearn import metrics
import numpy as np # for linear algebra
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler  # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier
from sklearn.cross_validation import train_test_split  # to split the data
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve, auc, roc_curve, roc_auc_score,classification_report
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("creditcard.csv",header=0)
sns.countplot("Class",data=data)

Count_Normal_transaction = len(data[data["Class"] == 0])  # normal transaction are repersented by 0
Count_Fraud_transaction = len(data[data["Class"] == 1])  # fraud by 1
print("Normal transactions",Count_Normal_transaction)
print("Fraud transactions",Count_Fraud_transaction)
Percentage_of_Normal_transaction = Count_Normal_transaction/(Count_Normal_transaction+Count_Fraud_transaction)
print("percentage of normal transaction is",Percentage_of_Normal_transaction*100)
Percentage_of_Fraud_transaction= Count_Fraud_transaction/(Count_Normal_transaction+Count_Fraud_transaction)
print("percentage of fraud transaction",Percentage_of_Fraud_transaction*100)


# amount related to the valid and fraud transaction
Fraud_transaction = data[data["Class"] == 1]
Normal_transaction = data[data["Class"] == 0]
plt.figure(figsize=(10, 6))
plt.subplot(121)
Fraud_transaction.Amount.plot.hist(title="Fraud Transaction")
plt.subplot(122)
Normal_transaction.Amount.plot.hist(title="Normal Transaction")
plt.show()
fraud_indices = np.array(data[data.Class == 1].index)
normal_indices = np.array(data[data.Class == 0].index)

def model(model, features_train, features_test, labels_train, labels_test):
    clf = model
    print("Training.....")
    clf.fit(features_train, labels_train.values.ravel())
    pred = clf.predict(features_test)
    probs = clf.predict_proba(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred)
    fig = plt.figure(figsize=(6, 3))  # to plot the graph
    print("TP", cnf_matrix[1, 1])  # no of fraud transaction which are predicted fraud
    print("TN", cnf_matrix[0, 0])  # no. of normal transaction which are predited normal
    print("FP", cnf_matrix[0, 1])  # no of normal transaction which are predicted fraud
    print("FN", cnf_matrix[1, 0])  # no of fraud Transaction which are predicted normal
    print("--------------Evaluation Criteria 1----------------")
    print("Accuracy is ", (cnf_matrix[1, 1] + cnf_matrix[0, 0])/(cnf_matrix[1, 1]+cnf_matrix[0, 0]+cnf_matrix[0, 1]
    + cnf_matrix[1, 0]))



    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()

    probs = probs[:, 1]
    auc = roc_auc_score(labels_test, probs)
    print('AUC Score is : %.3f' % auc)
    fpr, tpr, thresholds = roc_curve(labels_test, probs)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve Credit Card')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
    # show the plot
    plt.legend(loc='lower right')
    plt.savefig('CreditCardFraudTransactionDetectionRocRandomForest')
    plt.show()

    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test, pred))


def data_prepration(x_features,x_labels):  # preparing data for training and testing as we are going to use different data
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.3, random_state=0)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)


adsn = ADASYN(k=7,imb_threshold=0.6, ratio=6.5)
x = data
x_features = x.ix[:, x.columns != "Class"]
x_labels = x.ix[:, x.columns == "Class"]
os_data_X, os_data_y = adsn.fit_transform(x_features.values, [i[0] for i in x_labels.values])  # first oversampling
#  then splitting
data_train_X, data_test_X, data_train_y, data_test_y = data_prepration(os_data_X, os_data_y)
columns = x_features.columns
print(columns)


data_train_X = pd.DataFrame(data=data_train_X, columns=columns)
data_train_y = pd.DataFrame(data=data_train_y, columns=["Class"])
data_test_X = pd.DataFrame(data=data_test_X, columns=columns)
data_test_y = pd.DataFrame(data=data_test_y, columns=["Class"])
os_data_X = data_train_X
os_data_y = data_train_y
data_test_X_pandas = pd.DataFrame(data=data_test_X, columns=columns)
# now we divide our data into training and test data
# Call our method data preparation on our data-set


print("length of oversampled data is ",len(os_data_X))
print("Number of normal transaction in oversampled data",len(os_data_y[os_data_y["Class"]==0]))
print("No.of fraud transaction", len(os_data_y[os_data_y["Class"] == 1]))
print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["Class"] == 0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y["Class"] == 1])/len(os_data_X))

os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X['Amount'].values.reshape(-1, 1))
os_data_X.drop(["Time", "Amount"],axis=1,inplace=True)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].values.reshape(-1, 1))
data_test_X.drop(["Time", "Amount"],axis=1,inplace=True)


clf= RandomForestClassifier(n_estimators=100,random_state=0)
# train data using oversampled data and predict for the test data
model(clf,os_data_X,data_test_X,os_data_y,data_test_y)








