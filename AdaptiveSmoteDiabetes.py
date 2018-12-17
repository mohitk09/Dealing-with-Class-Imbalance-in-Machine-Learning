import pandas as pd  # to import csv and for data manipulation
import matplotlib.pyplot as plt  # to plot graph
import seaborn as sns  # for interactive graphs
from adasyn import ADASYN
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier  # Random forest classifier
from sklearn.cross_validation import train_test_split  # to split the data
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("diabetes.csv")
sns.countplot("Class", data=data)

Count_Normal = len(data[data["Class"] == 0])  # normal transaction are represented by 0
Count_People_With_Cancer = len(data[data["Class"] == 1])  # fraud by 1
print("Normal People", Count_Normal)
print("People_With_Cancer", Count_People_With_Cancer)
Percentage_of_Normal_People = Count_Normal/(Count_Normal+Count_People_With_Cancer)
print("Percentage of People who are normal",Percentage_of_Normal_People*100)
Percentage_of_People_with_cancer = Count_People_With_Cancer/(Count_Normal+Count_People_With_Cancer)
print("Percentage of People having cancer", Percentage_of_People_with_cancer*100)


def model(clf, features_train, features_test, labels_train, labels_test, type):
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
    print("Accuracy for " + type + " ", (cnf_matrix[1, 1] + cnf_matrix[0, 0])/(cnf_matrix[1, 1]+cnf_matrix[0, 0]+cnf_matrix[0, 1]
    +cnf_matrix[1, 0]))


    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    if type == 'RandomForest':
        probs = probs[:, 1]
        auc = roc_auc_score(labels_test, probs)
        print('AUC Score is : %.3f' % auc)
        fpr, tpr, thresholds = roc_curve(labels_test, probs)
        # plot no skill
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Curve Diabetes')
        # plot the roc curve for the model
        plt.plot(fpr, tpr, marker='.', label='ROC curve (area = %0.3f)' % auc)
        # show the plot
        plt.legend(loc = 'lower right')
        plt.savefig('DiabetesRocRandomForestAdaptiveSmote')
        plt.show()

    return classification_report(labels_test, pred)


def data_prepration(x_features, x_labels): # preparing data for training and testing as we are going to use different data
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.2, random_state=0)

    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train, x_features_test, x_labels_train, x_labels_test)


adsn = ADASYN(k=7,imb_threshold=0.7, ratio=1, random_state=0)


# now we can devided our data into training and test data
# Call our method data prepration on our dataset
x = data
x = x.dropna(axis=0,how='any')
x_features = x.ix[:, x.columns != "Class"]

x_labels = x.ix[:, x.columns == "Class"]
os_data_X, os_data_y = adsn.fit_transform(x_features.values, [i[0] for i in x_labels.values])
data_train_X, data_test_X, data_train_y, data_test_y = data_prepration(os_data_X, os_data_y)
columns = x_features.columns
print(columns)


data_train_X = pd.DataFrame(data=data_train_X, columns=columns )
data_train_y= pd.DataFrame(data=data_train_y, columns=["Class"])



print("Length of oversampled data is ", len(data_train_X))
print("Number of normal transcation in oversampled data", len(data_train_y[data_train_y["Class"] == 0]))
print("No.of fraud transcation", len(data_train_y[data_train_y["Class"]==1]))
print("Number of normal people in oversampled data is ",len(data_train_y[data_train_y["Class"] == 0])/len(data_train_X))
print("Number of people having diabetes in oversampled data is ", len(data_train_y[data_train_y["Class"]==1])/len(data_train_X))


list_of_results = []

# train data using oversampled data and predict for the test data
# Using three different classifiers
for i in range(0, 3):
    if i == 0:
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        type = "RandomForest"
    elif i == 1:
        clf = svm.SVC(kernel='linear',random_state=0, C=1, probability=True)
        type = "SVM"
    elif i == 2:
        clf = LogisticRegression(random_state=0)
        type = "LogisticRegression"
    result = model(clf, data_train_X, data_test_X, data_train_y, data_test_y,type)
    list_of_results.append(result)
print("\n----------Classification Report------------------------------------")
print(list_of_results[0])






