import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
warnings.filterwarnings("ignore")
%matplotlib inline
from tensorflow import keras
import scipy.stats as stats
from feature_engine.outlier_removers import Winsorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import tensorflow as tfs
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


df = pd.read_csv("insurance_claims.csv")
df.head()
df.shape
df.isnull().sum().sum()
df.info()
    df.authorities_contacted.unique()
df.collision_type.value_counts()

df["collision_type"] = np.where(df["collision_type"] == "?","missing",df["collision_type"])

df.property_damage.value_counts()

df["property_damage"] = np.where(df["property_damage"] == "?","missing",df["property_damage"])

df.police_report_available.value_counts()

df["police_report_available"] = np.where(df["police_report_available"] == "?","missing",df["police_report_available"])

df["policy_bind_date"] = pd.to_datetime(df["policy_bind_date"])
df["incident_date"] = pd.to_datetime(df["incident_date"])
df["date_diff"] = (df["incident_date"]-df["policy_bind_date"]).dt.days
df["make_model"] = df["auto_make"] + "_" + df["auto_model"]
df.drop(["policy_number","insured_zip","incident_location","policy_bind_date","incident_date","auto_make","auto_model"],
        axis = 1,inplace = True)
df.shape

plt.figure(figsize = (12,8))
g = sns.countplot(x="age",data=df,palette='hls')
g.set_title("different age groups", fontsize=20)
g.set_xlabel("age", fontsize=15)
g.set_ylabel("count", fontsize=20)

sns.countplot(df.fraud_reported)

df["policy_csl"].value_counts().plot(kind='barh', figsize=(20,10), title="ValueCount");

sns.set(style="ticks", color_codes=True)
fig, axes = plt.subplots(nrows = 4,ncols = 2,figsize = (25,15))
sns.countplot(x = "policy_state", data = df, hue = "fraud_reported",ax=axes[0][0])
sns.countplot(x = "insured_sex", data = df, hue = "fraud_reported",ax=axes[0][1])
sns.countplot(x = "insured_education_level", data = df, hue = "fraud_reported",ax=axes[1][0])
sns.countplot(x = "insured_hobbies", data = df, hue = "fraud_reported",ax=axes[1][1])
sns.countplot(x = "insured_relationship", data = df, hue = "fraud_reported",ax=axes[2][0])
sns.countplot(x = "incident_type", data = df,hue = "fraud_reported", ax=axes[2][1])
sns.countplot(x = "incident_severity", data = df,hue = "fraud_reported", ax=axes[3][0])
sns.countplot(x = "authorities_contacted", data = df,hue = "fraud_reported", ax=axes[3][1])
plt.show(fig)

fig, axarr = plt.subplots(5, 2, figsize=(20, 12))
sns.boxplot(y="months_as_customer",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[0][0])
sns.boxplot(y="age",x = "fraud_reported", hue = "fraud_reported",data = df , ax=axarr[0][1])
sns.boxplot(y="policy_deductable",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[1][0])
sns.boxplot(y="policy_annual_premium",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[1][1])
sns.boxplot(y="umbrella_limit",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[2][0])
sns.boxplot(y="capital-gains",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[2][1])
sns.boxplot(y="total_claim_amount",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[3][0])
sns.boxplot(y="injury_claim",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[3][1])
sns.boxplot(y="property_claim",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[4][0])
sns.boxplot(y="vehicle_claim",x = "fraud_reported", hue = "fraud_reported",data = df, ax=axarr[4][1])

df.drop(["umbrella_limit"],axis = 1, inplace = True)

def deeply_plots(df, variable_list):
    for variable in variable_list:
        print(variable)
        plt.figure(figsize=(16, 4))
        # histogram
        plt.subplot(1, 3, 1)
        sns.distplot(df[variable], bins=30)
        plt.title("Histogram")
        # QQ-plot
        plt.subplot(1, 3, 2)
        stats.probplot(df[variable], dist="norm", plot=plt)
        plt.ylabel("quantiles")
        # boxplot
        plt.subplot(1, 3, 3)
        sns.boxplot(y=df[variable])
        plt.title("Boxplot")
        plt.show()

deeply_plots(df, ["policy_annual_premium", "total_claim_amount", "property_claim", "vehicle_claim"])

wind = Winsorizer(distribution = 'skewed',
                  tail = 'both',
                  fold = 1.5,
                  variables=["policy_annual_premium","total_claim_amount","property_claim"])

wind.fit(df)
df = wind.transform(df)

deeply_plots(df, ["policy_annual_premium", "total_claim_amount", "property_claim"])

df["fraud_reported"] = np.where(df["fraud_reported"] == "N",1,0)


df["number_of_vehicles_involved"] = str(df["number_of_vehicles_involved"])
df["bodily_injuries"] = str(df["bodily_injuries"])
df["witnesses"] = str(df["witnesses"])

 
cat = ["policy_state","policy_csl","insured_sex","insured_education_level","insured_occupation",
       "insured_hobbies","insured_relationship","incident_type","collision_type","incident_severity",
       "authorities_contacted","incident_state","incident_city","number_of_vehicles_involved",
       "property_damage","bodily_injuries","witnesses","police_report_available","make_model"]

df_2 = pd.get_dummies(df[cat],drop_first=True)

df.drop(cat,axis = 1,inplace = True)
df_mod = pd.concat([df,df_2],axis = 1)

y = df_mod["fraud_reported"]
X = df_mod.drop(["fraud_reported"],axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

class FeatureSelector:

    def __init__(self, X_train):
        self.X_train = X_train

    def get_correlation_matrix(self):
        corr_matrix = self.X_train.corr()
        fig, ax = plt.subplots(figsize=(20, 15))
        ax = sns.heatmap(corr_matrix, annot=False, linewidths=0.5, fmt=".2f", cmap="YlGnBu")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

    @staticmethod
    def correlation(dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
                    
        return col_corr

    def get_corr_features_len(self):
        corr_features = self.correlation(self.X_train, 0.7)
        return corr_features

    def get_constant_features_len(self):
        constant_features = [
                feat for feat in self.X_train.columns if self.X_train[feat].std() == 0
            ]
        return len(constant_features)

    def get_duplicated_feat_len(self):
        duplicated_feat = []
        for i in range(0, len(self.X_train.columns)):
            if i % 10 == 0:
                print(i)
            col_1 = self.X_train.columns[i]
            for col_2 in self.X_train.columns[i + 1:]:
                if self.X_train[col_1].equals(self.X_train[col_2]):
                    duplicated_feat.append(col_2)

        return len(set(duplicated_feat))
    
    def get_roc_values(self):
        roc_values = []
        for feature in self.X_train.columns:
            clf = RandomForestClassifier()
            clf.fit(self.X_train[feature].fillna(0).to_frame(), y_train)
            y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
            roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

        roc_values = pd.Series(roc_values)
        roc_values.index = self.X_train.columns
        roc_values.sort_values(ascending=False)

        roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))
        return roc_values

feature_selector = FeatureSelector(X_train)
feature_selector.get_correlation_matrix()

feature_selector.get_corr_features_len()
corr_features = feature_selector.get_corr_features_len()

X_train.drop(corr_features, axis = 1,inplace = True)
X_test.drop(corr_features, axis = 1,inplace = True)
X_train.shape,X_test.shape

feature_selector.get_constant_features_len()
feature_selector.get_duplicated_feat_len()

roc_values = feature_selector.get_roc_values()
roc_values

len(roc_values[roc_values > 0.5])

drop_list = roc_values[roc_values < 0.5].index

X_train.drop(drop_list,axis = 1,inplace = True)
X_test.drop(drop_list,axis = 1,inplace = True)

X_train.shape,X_test.shape

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)

#preprocessing ve kategorik dönüşümleri yapılmış olması gerekli
#overfit i önlemek için train verisi üzerinde çalışılmalı
#ilk olarak korele olan değişkenler elenir
sfs1 = SFS(LogisticRegression(n_jobs = -1),
           k_features = 15,
           forward = False,
           floating = False,
           verbose = 2,
           scoring = "roc_auc",
           cv = 3)
#stepbackword yaptıgımız için forward kısmını False yaptık
sfs1 = sfs1.fit(np.array(X_train.fillna(0)),y_train)
selected_feat = X_train.columns[list(sfs1.k_feature_idx_)]
selected_feat

#logistic
logistic =  LogisticRegression(random_state = 0)
logistic.fit(X_train_df[selected_feat], y_train)

#knn
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train_df[selected_feat], y_train)

#svm
svm = SVC(kernel = 'linear', random_state = 0,probability = True)
svm.fit(X_train_df[selected_feat], y_train)

modellist = [logistic,knn,svm]
for model in modellist:
    print(model.score(X_train_df[selected_feat],y_train))
    
mods   = ['Logistic Regression','KNN Classifier','SVM Classifier Linear']
for i,k in zip(modellist,mods):
    predictions = i.predict(X_test_df[selected_feat])
    print("{}".format(k))
    print(classification_report(y_test,predictions))
    
model = tfs.keras.models.Sequential((tfs.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', 
                                                               activation = 'relu', input_dim = 15),
                                         tfs.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', 
                                                                activation = 'relu'),
                                         tfs.keras.layers.Dense(units = 10, kernel_initializer = 'uniform', 
                                                                activation = 'relu'),
                                         tfs.keras.layers.Dense(units = 6, kernel_initializer = 'uniform', 
                                                                activation = 'relu'),
                                         tfs.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', 
                                                               activation = 'sigmoid')))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train_df[selected_feat],y_train,batch_size = 100, epochs = 500)

y_pred = model.predict(X_test_df[selected_feat])
y_pred = (y_pred > 0.5)

score = accuracy_score(y_test,y_pred)
score

param_grid={"C": [0.001,0.01,0.1,1,10,100,1000], "penalty":["l1","l2"]}
clf=LogisticRegression()
logreg_cv=GridSearchCV(clf,param_grid,cv=10)

logreg_cv.fit(X_train_df[selected_feat],y_train)

print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)

y_prediction = logreg_cv.predict(X_test_df[selected_feat])

cm = confusion_matrix(y_test,y_prediction)

sns.heatmap(cm,annot=True)
plt.savefig('h.png')

accuracy_score(y_test,y_prediction)