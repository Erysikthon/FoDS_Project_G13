from Data_Preparation import *
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("data/kzp-2008-2020-timeseries.csv", encoding="latin-1")

####################### DATA OVERVIEW ##########################################################################
print('Shape of initial data set: ', df.shape)
print('Data types;', df.dtypes.unique())

#Total Missing Data
total_missing = df.isnull().sum().sum()  # Count total missing values
total_entries = df.size  # Total number of entries (rows * columns)
missing_percentage = (total_missing / total_entries) * 100  # Calculate missing percentage

print(f"Missing Data Percentage: {missing_percentage:.2f}%")

################################# ADJUSTED DATA SET #############################################

data = df.copy()

#LABEL AND FEATURES DECLARATION
label = "FiErg"

features = ["KT","Inst", "Adr",  "Ort", "Typ", "RWStatus", "Akt", "SL", "WB", "AnzStand","SA","PtageStatT","AustStatT","NeugStatT","Ops","Gebs","CMIb","CMIn",
            "pPatWAU","pPatWAK", "pPatLKP","pPatHOK","PersA","PersP","PersMT","PersT","PersAFall","PersPFall","PersMTFall","PersTFall","AnzBelA","AnzBelP (nur ab KZP2010)"]


data["JAHR"] = data["JAHR"].astype("object")

#New data declaration
data = data[features + [label] + ["JAHR"]]

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

#Dropping SA
data = data.drop(columns=["SA"])
features.remove("SA")


#DECLARING NUM / CAT FEATURES
num_features = data[features].select_dtypes("number").columns
cat_features = data[features].select_dtypes(exclude=["number"]).columns

print('Unique variables per categorical column:')
for col in cat_features:  # Iterate through each categorical column
    print(f'{col}: {data[col].nunique()} unique values')

######################### MISSING DATA ANALYSIS #########################################################################

def analyze_missing_patterns(df):

    # Missing data matrix (True/False)
    missing_matrix = df.isnull()

    # Visualizing missing data patterns
    plt.figure(figsize=(20, 14))
    sns.heatmap(missing_matrix, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Pattern')
    plt.xlabel('Variables (Columns)')
    plt.ylabel('Observations (Rows)')
    plt.show()

    # Checking for relationships between missing values
    missing_corr = missing_matrix.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm')
    plt.title('Correlation of Missing Values')
    plt.show()

    # Checking if missingness in one variable depends on other variables
    for col in df.columns:
        if df[col].isnull().any():
            # Comparing means of other numeric columns when this column is missing vs not missing
            missing_mask = df[col].isnull()
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            print(f"\nAnalyzing missingness in {col}:")
            for other_col in numeric_cols:
                if other_col != col:
                    not_missing_mean = df[~missing_mask][other_col].mean()
                    missing_mean = df[missing_mask][other_col].mean()
                    print(f"{other_col}:")
                    print(f"  Mean when {col} is not missing: {not_missing_mean:.2f}")
                    print(f"  Mean when {col} is missing: {missing_mean:.2f}")


analyze_missing_patterns(data)


#Percentage of missing values in Label column
missing_percentage_label = (data[label].isnull().sum() / len(data[label])) * 100
print("Missing percentage Label:\n", missing_percentage_label)

#Percentage of missing values per categorical column
missing_percentage_cat = (data[cat_features].isnull().sum() / len(data[cat_features])) * 100
print("Missing percentage Categorical columns:\n", missing_percentage_cat)

#Percentage of missing values per numeric column
missing_percentage_num = (data[num_features].isnull().sum() / len(data[num_features])) * 100
print("Missing percentage numeric columns:\n", missing_percentage_num)



#Dropping Missing Data in label column
data = data.dropna(subset=['FiErg'])


#Transforming label to 0/1
data[label] = data[label].apply(lambda x: 0 if x < 0 else 1)

#Total Missing Data
total_missing = data.isnull().sum().sum()  # Count total missing values
total_entries = data.size  # Total number of entries (rows * columns)
missing_percentage = (total_missing / total_entries) * 100  # Calculate missing percentage

print(f"Missing Data Percentage (new data): {missing_percentage:.2f}%")


#LABEL BALANCE CHECK
print(data[label].value_counts(normalize=True))




#################################### Correlations  ###########################################################
print("KT dtype:", df["KT"].dtype)
print("EtSubv dtype:", df["EtSubv"].dtype)


def correlation_ratio(categorical, numeric):
    categories = pd.unique(categorical)
    n = len(numeric)

    # Calculate means per category
    cat_means = np.array([numeric[categorical == cat].mean() for cat in categories])
    cat_counts = np.array([sum(categorical == cat) for cat in categories])

    # Calculate total mean
    total_mean = numeric.mean()

    # Calculate weighted sum of squared deviations from total mean
    numerator = sum(count * (mean - total_mean) ** 2
                    for count, mean in zip(cat_counts, cat_means))

    # Calculate total sum of squared deviations
    denominator = sum((numeric - total_mean) ** 2)

    # Return correlation ratio
    return np.sqrt(numerator / denominator)


#################################### KT - EtSubv Correlation ###########################
correlations = {}

clean_df = df.dropna(subset=['KT', 'EtSubv'])

# correlation ratio using the whole dataset directly
eta = correlation_ratio(clean_df['KT'], clean_df['EtSubv'])

print("Overall correlation ratio between cantons and subventions:", eta)

# distribution of subventions by canton
plt.figure(figsize=(12, 8))
sns.boxplot(data=clean_df, x='KT', y='EtSubv')
plt.title('Distribution of Subventions by Canton')
plt.xticks(rotation=45)
plt.xlabel('Canton')
plt.ylabel('Subventions (EtSubv)')
plt.tight_layout()
plt.show()





############################# KT - FiErg Correlation ####################################
correlations = {}

clean_df = df.dropna(subset=['KT', 'FiErg'])

# Calculating correlation ratio using the whole dataset directly
eta = correlation_ratio(clean_df['KT'], clean_df['FiErg'])

print("Overall correlation ratio between cantons and yearly return:", eta)

#distribution of FiErg by canton
plt.figure(figsize=(12, 8))
sns.boxplot(data=clean_df, x='KT', y='FiErg')
plt.title('Distribution of FiErg by KT')
plt.xticks(rotation=45)
plt.xlabel('KT')
plt.ylabel('FiErg)')
plt.tight_layout()
plt.show()

########################################## Correlation EtSubv - FiErg
#Correlation measures
pearson_corr = clean_df['EtSubv'].corr(clean_df['FiErg'])
spearman_corr = clean_df['EtSubv'].corr(clean_df['FiErg'], method='spearman')

print(f"Pearson correlation coefficient (EtSubv-FiErg): {pearson_corr:.3f}")
print(f"Spearman correlation coefficient (EtSubv-FiErg): {spearman_corr:.3f}")

# Scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(clean_df['EtSubv'], clean_df['FiErg'], alpha=0.5)
plt.xlabel('Subventions (EtSubv)')
plt.ylabel('Financial Results (FiErg)')
plt.title(f'Correlation between Subventions and Financial Results\nPearson r = {pearson_corr:.3f}')

# trend line
z = np.polyfit(clean_df['EtSubv'], clean_df['FiErg'], 1)
p = np.poly1d(z)
plt.plot(clean_df['EtSubv'], p(clean_df['EtSubv']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

