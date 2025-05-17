import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

df = pd.read_csv("data/kzp-2008-2020-timeseries.csv", encoding="latin-1")

#DATA OVERVIEW
print('Shape of initial Data Set: ', df.shape)
print('Data types;', df.dtypes.unique())

#SPECIFIY YEAR
data =df[df["JAHR"]==2011].copy()

#ALL YEARS
#data = df.copy()

#LABEL AND FEATURES DECLARATION
label = "FiErg"

features = ["JAHR","KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]

#New data declaration
data = data[features + [label]]
print(data.shape)

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


#MISSING DATA OVERVIEW
#Percentage of missing values in label column
missing_percentage_label = (data[label].isnull().sum() / len(data[label])) * 100
print("Missing percentage Label\n", missing_percentage_label)

#Percentage of missing values per categorical column
missing_percentage_cat = (data[cat_features].isnull().sum() / len(data[cat_features])) * 100
print("Missing percentage Categorical columns\n", missing_percentage_cat)

#Percentage of missing values per numeric column
missing_percentage_num = (data[num_features].isnull().sum() / len(data[num_features])) * 100
print("Missing percentage numeric columns\n", missing_percentage_num)



#Dropping Missing Data in label column
data = data.dropna(subset=['FiErg'])


#Transforming label to 0/1
data[label] = data[label].apply(lambda x: 0 if x < 0 else 1)


print('Shape of modified data set;', data.shape)

#LABEL BALANCE CHECK
print(data[label].value_counts(normalize=True))

