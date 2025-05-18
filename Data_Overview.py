from Data_Preparation import *

#EtSubv
#correlation kt subventionen

df = pd.read_csv("data/kzp-2008-2020-timeseries.csv", encoding="latin-1")

####################### DATA OVERVIEW ##########################################################################
print('Shape of initial data set: ', df.shape)
print('Data types;', df.dtypes.unique())

#SPECIFIC YEAR
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


#MISSING DATA OVERVIEW
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
#data[label] = data[label].apply(lambda x: 0 if x < 0 else 1)


print('Shape of modified data set;', data.shape)

#LABEL BALANCE CHECK
print(data[label].value_counts(normalize=True))




#################################### Correlation KT - EtSubv ###########################################################
print("KT dtype:", df["KT"].dtype)
print("EtSubv dtype:", df["EtSubv"].dtype)
print("Any null values in KT?", df["KT"].isnull().any())
print("Any null values in EtSubv?", df["EtSubv"].isnull().any())

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


#################################### KT - EtSubv ################################
correlations = {}

# First, clean the data by removing rows with null values in either column
clean_df = df.dropna(subset=['KT', 'EtSubv'])

# Calculate correlation ratio using the whole dataset directly
eta = correlation_ratio(clean_df['KT'], clean_df['EtSubv'])

print("\nOverall correlation ratio between cantons and subventions:", eta)

# For better analysis, let's look at the distribution of subventions by canton
plt.figure(figsize=(12, 8))
sns.boxplot(data=clean_df, x='KT', y='EtSubv')
plt.title('Distribution of Subventions by Canton')
plt.xticks(rotation=45)
plt.xlabel('Canton')
plt.ylabel('Subventions (EtSubv)')
plt.tight_layout()
plt.show()





################ KT - FiErg
correlations = {}

# First, clean the data by removing rows with null values in either column
clean_df = df.dropna(subset=['KT', 'FiErg'])

# Calculate correlation ratio using the whole dataset directly
eta = correlation_ratio(clean_df['KT'], clean_df['FiErg'])

print("\nOverall correlation ratio between cantons and subventions:", eta)

# For better analysis, let's look at the distribution of subventions by canton
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

print(f"Pearson correlation coefficient: {pearson_corr:.3f}")
print(f"Spearman correlation coefficient: {spearman_corr:.3f}")

# Scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(clean_df['EtSubv'], clean_df['FiErg'], alpha=0.5)
plt.xlabel('Subventions (EtSubv)')
plt.ylabel('Financial Results (FiErg)')
plt.title(f'Correlation between Subventions and Financial Results\nPearson r = {pearson_corr:.3f}')

# Add trend line
z = np.polyfit(clean_df['EtSubv'], clean_df['FiErg'], 1)
p = np.poly1d(z)
plt.plot(clean_df['EtSubv'], p(clean_df['EtSubv']), "r--", alpha=0.8)

plt.tight_layout()
plt.show()

