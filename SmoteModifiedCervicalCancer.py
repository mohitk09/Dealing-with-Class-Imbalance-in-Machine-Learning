from sklearn.cross_validation import train_test_split
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


data = pd.read_excel("Cervical_Cancer.csv")
sns.countplot("Class", data=data)  # Class represents biopsy
Count_Normal = len(data[data["Class"] == 0])  # normal People are represented by 0
Count_People_With_Cancer = len(data[data["Class"] == 1])  # People with cancer by 1
print("Normal People", Count_Normal)
print("People_With_Cancer", Count_People_With_Cancer)
Percentage_of_Normal_People = Count_Normal/(Count_Normal+Count_People_With_Cancer)
print("Percentage of People who are normal",Percentage_of_Normal_People*100)
Percentage_of_People_with_cancer = Count_People_With_Cancer/(Count_Normal+Count_People_With_Cancer)
print("Percentage of People having cancer", Percentage_of_People_with_cancer*100)


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


def data_prepration(x_features, x_labels): # preparing data for training and testing as we are going to use different data
    x_features_train, x_features_test, x_labels_train, x_labels_test = train_test_split(x_features, x_labels, test_size= 0.3)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train, x_features_test, x_labels_train, x_labels_test)


def preprocessing_of_data():
    data = pd.read_excel("Cervical_Cancer.csv")
    x = data
    x = x.dropna(axis=0, how='any')
    x_features= x.ix[:, x.columns != "Class"]
    x = x_features.values  #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    total = 0
    for i in range(0, 30):
        mean = df[i].mean()
        total = total+mean
    k_value = total/30
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



N = 100
k = 7
x = data
x = x.dropna(axis=0, how='any')
os = smote_modified(np.array(x), N, k=k)
smote = os.over_sampling()
smote = pd.DataFrame(smote)
print(smote)
os_data_X = smote.ix[:, smote.columns != 30]
os_data_y = smote.ix[:, smote.columns == 30]
print(os_data_X)
print(os_data_y)
for value in os_data_y.values:
    if value > 0.5:
        value = 1
    else:
        value = 0

data_train_X, data_test_X, data_train_y, data_test_y = data_prepration(os_data_X, os_data_y)
columns = os_data_X.columns
data_train_X = pd.DataFrame(data=data_train_X, columns=columns )
data_train_y= pd.DataFrame(data=data_train_y, columns=["Class"])
print("Length of oversampled data is ", len(data_train_X))
print("Number of normal transcation in oversampled data", len(data_train_y[data_train_y["Class"] == 0]))
print("No.of fraud transcation", len(data_train_y[data_train_y["Class"]==1]))
print("Proportion of Normal data in oversampled data is ",len(data_train_y[data_train_y["Class"] == 0])/len(data_train_X))
print("Proportion of fraud data in oversampled data is ",len(data_train_y[data_train_y["Class"]==1])/len(data_train_X))

list_of_results = []

# train data using oversampled data and predict for the test data
clf = RandomForestClassifier(n_estimators=100, random_state=0)
type = "RandomForest"
result = model(clf, data_train_X, data_test_X, data_train_y, data_test_y)
list_of_results.append(result)
print("\n----------Classification Report------------------------------------")
print(list_of_results)
