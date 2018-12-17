import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')
from sklearn.cross_validation import train_test_split


def model(model, features_train, features_test, labels_train, labels_test):
    clf = model
    print("Training.....")
    clf.fit(features_train, labels_train.values.ravel())
    pred = clf.predict(features_test)
    cnf_matrix = confusion_matrix(labels_test, pred)
    print("the recall for this model is :", cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]))
    fig = plt.figure(figsize=(6, 3))  # to plot the graph
    print("TP", cnf_matrix[1, 1,])  # no of fraud transaction which are predicted fraud
    print("TN", cnf_matrix[0, 0])  # no. of normal transaction which are predited normal
    print("FP", cnf_matrix[0, 1])  # no of normal transaction which are predicted fraud
    print("FN", cnf_matrix[1, 0])  # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix, cmap="coolwarm_r", annot=True, linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(labels_test, pred))


def data_prepration(x): # preparing data for training and testing as we are going to use different data
    x_features = x.ix[:,x.columns != "Class"]
    x_labels = x.ix[:,x.columns == "Class"]
    x_features_train, x_features_test, x_labels_train, x_labels_test = train_test_split(x_features, x_labels, test_size= 0.3)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train, x_features_test, x_labels_train, x_labels_test)


def preprocessing_of_data():
    data = pd.read_csv("creditcard.csv",header = 0)
    x = data
    x_features= x.ix[:,x.columns != "Class"]
    x = x_features.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    total = 0
    for i in range(1, 29):
        mean = df[i].mean()
        total = total+mean
    k_value = total/28
    return k_value


class smote_modified:

    def __init__(self, samples, N=10, k=5):
        self.n_samps, self.n_attrs = samples.shape
        self.N = N
        self.k = k
        self.samples = samples
        self.k_value = preprocessing_of_data()
        print("K value is ",self.k_value)

    def over_sampling(self):

        self.n_synth = int((self.N / 100) * self.n_samps)  # Randomize minority class samples

        rand_indexes = np.random.permutation(self.n_samps)
        if self.N > 100:
            self.N = np.ceil(self.N / 100)
            self.N = int(self.N)
            for i in range(self.N - 1):
                rand_indexes = np.append(rand_indexes, np.random.permutation(self.n_samps))

        self.syntethic = np.zeros((self.n_synth, self.n_attrs));
        self.newindex = 0

        nearest_k = NearestNeighbors(n_neighbors=self.k).fit(self.samples)

        # for i in range (0, self.n_samps-1):
        for i in rand_indexes[:self.n_synth]:
            nnarray = nearest_k.kneighbors([self.samples[i]], return_distance=False)[0]
            self.__populate(i, nnarray)

        return self.syntethic

    def __populate(self, i, nnarray):
        ## Choose a random number between 0 and k
        nn = np.random.randint(0, self.k)
        while nnarray[nn] == i:
            nn = np.random.randint(0, self.k)

        dif = self.samples[nnarray[nn]] - self.samples[i]
        #gap = np.random.rand(1, self.n_attrs)
        self.syntethic[self.newindex] = self.samples[i] + self.k_value*dif
        self.newindex += 1
        return


data = pd.read_csv("creditcard.csv", header = 0)
N = 200
k = 5
data_train_X, data_test_X, data_train_y, data_test_y = data_prepration(data)
dropping_column = pd.DataFrame(data_train_X)
dropping_column.drop(["Time"],axis=1, inplace=True)
columnsx = dropping_column.columns
columnsy = data_train_y.columns

new_samples = [[0 for i in range(0, 30)] for j in range(0, len(data_train_X))]
for i in range(0, len(data_train_X)):
    for j in range(0, 29):
        new_samples[i][j] = data_train_X.values[i][j+1]
    new_samples[i][29] = data_train_y.values[i][0]
os = smote_modified(np.array(new_samples), N, k=k)
smote = os.over_sampling()
smote = pd.DataFrame(smote)
print(smote)
os_data_X = smote.drop([29], axis=1)
os_data_y = smote.drop([i for i in range(0, 29)], axis=1)
print(os_data_X)
print(os_data_y)
# we can Check the numbers of our data
print("length of oversampled data is ", len(smote))
print("Number of normal transcation in oversampled data", len(os_data_y[os_data_y[29] == 0]))
print("No.of fraud transcation",len(os_data_y[os_data_y[29] == 1]))
print("Proportion of Normal data in oversampled data is ", len(os_data_y[os_data_y[29] == 0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ", len(os_data_y[os_data_y[29] == 1])/len(os_data_X))

os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X[28].values.reshape(-1, 1))
os_data_X.drop([28],axis=1, inplace=True)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X["Amount"].values.reshape(-1, 1))
data_test_X.drop(["Time", "Amount"], axis=1, inplace=True)

for i in range(0, len(os_data_y)):
    os_data_y.values[i] = int(os_data_y.values[i])

clf= RandomForestClassifier(n_estimators=100)
# train data using oversampled data and predict for the test data
model(clf,os_data_X,data_test_X,os_data_y,data_test_y)