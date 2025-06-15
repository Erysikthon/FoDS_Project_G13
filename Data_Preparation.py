
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer



df = pd.read_csv("data/kzp-2008-2020-timeseries.csv", encoding="latin-1")

data = df.copy()

#LABEL AND FEATURES DECLARATION
label = "FiErg"

features = ["KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]

# JAHR data type conversion
data["JAHR"] = data["JAHR"].astype("object")


#New data declaration
data = data[features + [label] + ["JAHR"]]

#print("New data set shape:", data.shape)

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

#Dropping SA
data = data.drop(columns=["SA"])
features.remove("SA")


#DECLARING NUM / CAT FEATURES
num_features = data[features].select_dtypes("number").columns
cat_features = data[features].select_dtypes(exclude=["number"]).columns


#Dropping Missing Data in label column
data = data.dropna(subset=['FiErg'])


#Transforming label to 0/1
data[label] = data[label].apply(lambda x: 0 if x < 0 else 1)


#print('Shape of modified data set;', data.shape)



