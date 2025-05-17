import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/kzp-2008-2020-timeseries.csv", encoding="latin-1")

#SPECIFIY YEAR
current_year = "all"       #change this for individual year analysis, for all year -> "all"

if current_year != "all":
    data =df[df["JAHR"]==current_year].copy()

else:
    data = df.copy()

#LABEL AND FEATURES DECLARATION
label = "FiErg"

features = ["KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]

#for all years
#features.append("JAHR")
data["JAHR"] = data["JAHR"].astype("object")


#New data declaration
if current_year == "all":
    data = data[features + [label] + ["JAHR"]]
else:
    data = data[features + [label]]

print("New data set shape:", data.shape)

#DROPPING COLUMNS
#Dropping Adr
data = data.drop(columns=["Adr"])
features.remove("Adr")

#Dropping Inst
data = data.drop(columns=["Inst"])
features.remove("Inst")

#Dropping Ort
data = data.drop(columns=["Ort"])
features.remove("Ort")

#Dropping SA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
data = data.drop(columns=["SA"])
features.remove("SA")

#Dropping Columns with 100% Missing Data
#columns to drop
columns_to_drop = [i for i in features if data[i].isna().sum() == len(data[i])]

#Dropping the columns
data = data.drop(columns=columns_to_drop)

#Removing dropped columns from features
features = [i for i in features if i not in columns_to_drop]

#Output dropped columns
for col in columns_to_drop:
    print(f"Dropping {col} due to 100% Missing Data")


#DECLARING NUM / CAT FEATURES
num_features = data[features].select_dtypes("number").columns
cat_features = data[features].select_dtypes(exclude=["number"]).columns

print('Unique variables per categorical column:')
for col in cat_features:  # Iterate through each categorical column
    print(f'{col}: {data[col].nunique()} unique values')


#Dropping Missing Data in label column
data = data.dropna(subset=['FiErg'])


#Transforming label to 0/1
data[label] = data[label].apply(lambda x: 0 if x < 0 else 1)


print('Shape of modified data set;', data.shape)

#LABEL BALANCE CHECK
print(data[label].value_counts(normalize=True))

####################### NEW SPLITTING DATA ####################################################
X = data[features]
y = data[label]

yeares = data["JAHR"]

#for all years
if current_year == "all":
    stratify_col = y.astype(str) + "_" + yeares.astype(str)

else:
    stratify_col = y

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify= stratify_col)  #for all years also stratify years


############################ NEW MISSING DATA HANDLING ###########################################

#Filling Categorical Columns
for n in cat_features:
    X_train[n] = X_train[n].fillna("NA")
    X_test[n] = X_test[n].fillna("NA")

#Filling Numeric Columns
for i in X_train[num_features]:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_train[i].mean())      # To ensure consistency with X_train's mean


X_train[cat_features] = X_train[cat_features].astype('category')
X_test[cat_features] = X_test[cat_features].astype('category')

######################### NEW ONE HOT ENCODING ##################################################################
X_train = pd.get_dummies(X_train, columns=cat_features, drop_first=True)

X_test = pd.get_dummies(X_test, columns=cat_features, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

####################### NEW SCALING (after splitting!!) ###############################################################
sc = StandardScaler()
X_train[num_features] = sc.fit_transform(X_train[num_features])
X_test[num_features]  = sc.transform(X_test[num_features])