import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv(r"C:\Hello Files\Matplotlib and Seaborn\house_prices.csv")

print("Shape:", df.shape)
print(df.info())
print(df.head())

# 2. Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# 3. Missing value percentage
print("\nMissing Value %:")
missing_pct = df.isnull().mean() * 100
print(missing_pct)

# Fill missing values with median
df.fillna(df.median(numeric_only=True), inplace=True)

# 4. Distribution plots
numeric_cols = df.select_dtypes(include=np.number).columns

plt.figure()
plt.hist(df[col], bins=20)
plt.title(f"Histogram of {col}")
plt.xlabel(col)
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.boxplot(df[col], vert=False)
plt.title(f"Boxplot of {col}")
plt.show()

# 5. IQR Outlier Detection
outlier_summary = {}

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df[col] < lower) | (df[col] > upper)]
outlier_summary[col] = len(outliers)

# 6. Outlier flag column
df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)

# 7. Handle outliers (capping)
df[col] = np.where(df[col] < lower, lower,
                    np.where(df[col] > upper, upper, df[col]))

print("\nOutlier Count Per Column:")
print(outlier_summary)

# 8. Correlation Matrix
corr = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(6,4))
plt.imshow(corr)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.show()

# 9. Export cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nCleaned dataset exported successfully!")