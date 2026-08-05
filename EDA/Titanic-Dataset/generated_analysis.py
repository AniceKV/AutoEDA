DATA_FILEPATH = r'C:\Users\Anish Kumar Verma\PycharmProjects\AutoEDA\test_data\Titanic-Dataset.csv'

import pandas as pd
import numpy as np
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load pre-computed metadata profile
with open('metadata_profile.json', 'r') as f:
    metadata = json.load(f)

# Load the dataset from the global path variable
df = pd.read_csv(DATA_FILEPATH)

# ---------------------------------------------------------------------------
# 1. Smart Imputation Strategy
# ---------------------------------------------------------------------------
missing_cols = list(metadata.get('missing_values_summary', {}).keys())

for col in missing_cols:
    if col not in df.columns:
        continue

    if df[col].dtype in ['float64', 'int64']:
        # Numeric column – use skewness to decide median vs mean
        col_skew = df[col].skew()
        if abs(col_skew) > 1:
            fill_val = df[col].median()
        else:
            fill_val = df[col].mean()
        df[col] = df[col].fillna(fill_val)
    else:
        # Categorical / string column – mode or placeholder
        if col == 'Cabin':
            df[col] = df[col].fillna('Unknown')
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

# ---------------------------------------------------------------------------
# 2. Outlier Profiling (IQR-based)
# ---------------------------------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
key_numeric_cols = [c for c in numeric_cols if c not in ['PassengerId', 'Survived']]

outlier_rates = {}
for col in key_numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    rate = (n_outliers / len(df)) * 100
    outlier_rates[col] = round(rate, 2)
    print(f"Outlier rate in {col}: {rate:.2f}%")

# ---------------------------------------------------------------------------
# 3. Domain-Specific Feature Engineering
# ---------------------------------------------------------------------------
# Feature 1: Total family size aboard
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Feature 2: Extract title from Name
def extract_title(name):
    match = re.search(r'\b([A-Za-z]+)\.', name)
    return match.group(1) if match else 'Unknown'

df['Title'] = df['Name'].apply(extract_title)

# Collapse rare titles into a single category
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# Feature 3: Binary indicator for traveling alone
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ---------------------------------------------------------------------------
# 4. Statistical Hypothesis Testing against Survived
# ---------------------------------------------------------------------------
target = 'Survived'
test_results = {}

# Categorical predictors → Chi-Square test
categorical_features = ['Sex', 'Embarked', 'Pclass', 'Title']
for feat in categorical_features:
    contingency = pd.crosstab(df[feat], df[target])
    chi2, p_val, dof, ex = stats.chi2_contingency(contingency)
    test_results[feat] = {
        'test': 'Chi-Square',
        'statistic': round(chi2, 4),
        'p_value': round(p_val, 6)
    }
    print(f"Chi-Square test for {feat}: chi2={chi2:.4f}, p-value={p_val:.6f}")

# Numerical predictors → Welch's T-test
numerical_features = ['Age', 'Fare', 'FamilySize']
for feat in numerical_features:
    group0 = df.loc[df[target] == 0, feat].dropna()
    group1 = df.loc[df[target] == 1, feat].dropna()
    ttest_res = stats.ttest_ind(group0, group1, equal_var=False)
    t_stat = ttest_res.statistic
    p_val = ttest_res.pvalue
    test_results[feat] = {
        'test': 'Independent T-test (Welch)',
        'statistic': round(t_stat, 4),
        'p_value': round(p_val, 6)
    }
    print(f"T-test for {feat}: t-statistic={t_stat:.4f}, p-value={p_val:.6f}")

# ---------------------------------------------------------------------------
# 5. Advanced Visualizations
# ---------------------------------------------------------------------------
# 5a. Pearson correlation heatmap of all numeric variables
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr(method='pearson')

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Pearson Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
plt.close()

# 5b. Segmented violin plots (Age & Fare vs Survived)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.violinplot(x=df[target], y=df['Age'], ax=axes[0], palette='Set2')
axes[0].set_title('Age Distribution by Survival')
axes[0].set_xlabel('Survived')
axes[0].set_ylabel('Age')

sns.violinplot(x=df[target], y=df['Fare'], ax=axes[1], palette='Set2')
axes[1].set_title('Fare Distribution by Survival')
axes[1].set_xlabel('Survived')
axes[1].set_ylabel('Fare')

plt.tight_layout()
plt.savefig('target_interactions.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Modeling Blueprint JSON
# ---------------------------------------------------------------------------
significant_predictors = [
    feat for feat, res in test_results.items() if res['p_value'] < 0.05
]

metrics_dict = {
    'engineered_features': [
        {
            'name': 'FamilySize',
            'description': 'Total number of family members aboard (SibSp + Parch + 1)',
            'type': 'int64'
        },
        {
            'name': 'Title',
            'description': 'Extracted title from passenger name (Mr, Mrs, Miss, Master, Rare)',
            'type': 'str'
        },
        {
            'name': 'IsAlone',
            'description': 'Binary indicator for passengers traveling alone (FamilySize == 1)',
            'type': 'int64'
        }
    ],
    'outlier_rates': outlier_rates,
    'significant_predictors': significant_predictors,
    'statistical_tests': test_results,
    'modeling_strategy': (
        "Given the dataset size (891 samples) and binary target (Survived), a regularized logistic regression "
        "or gradient boosting classifier (e.g., XGBoost / LightGBM) is recommended. Key features to include are "
        "Sex, Title, Pclass, FamilySize, Age (imputed), Fare (log-transformed due to skewness), and IsAlone. "
        "Perform 5-fold stratified cross-validation. Use class weighting or SMOTE to handle class imbalance "
        "(survival rate ~38%). Evaluate with ROC-AUC and accuracy. Feature importance from tree-based models "
        "can provide further insights. The statistical tests indicate Sex, Pclass, Title, and Fare are significant "
        "predictors; Age and FamilySize may also contribute."
    )
}

with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent=2)

print("EDA and feature engineering complete. Files saved: correlation_matrix.png, target_interactions.png, metrics.json")