# Executive Summary Report: Zoo Animal Attribute EDA

## 1. Executive Overview

The automated EDA was performed on `zoo_data-1.csv`, a small animal-attribute dataset containing 101 observations. The available data consists primarily of binary biological and behavioral indicators, together with an animal name field and several engineered scores reported in the metrics artifact.

The principal conclusions are:

- No explicit supervised target or class label was detected.
- The dataset is complete with respect to missing values in all profiled columns.
- The feature space is dominated by binary indicators with varying prevalence, including several highly imbalanced traits.
- Strong correlations exist among biologically related attributes, particularly between:
  - `eggs` and `milk`: -0.939
  - `milk` and `terrestrial_mammalian_score`: 0.915
  - `locomotion_structure_score` and `tail`: 0.903
  - `hair` and `milk`: 0.879
- The engineered features are highly related to their source variables, which is expected given their construction. This should be considered when interpreting feature importance or using linear models.
- There is a metadata consistency issue: the declared dataset shape is 101 rows by 17 columns, while the metrics artifact also reports three engineered columns, resulting in a 20-column summary.
- Predictive modeling cannot be finalized until a target variable is supplied and its intended meaning is confirmed.

Overall, the dataset is suitable for descriptive biological pattern analysis and, after target confirmation, for small-sample classification experiments using interpretable and leakage-safe modeling practices.

---

## 2. Dataset Profile

### 2.1 Dataset dimensions and structure

| Property | Reported value |
|---|---:|
| Dataset name | `zoo_data-1.csv` |
| Rows | 101 |
| Columns in primary profile | 17 |
| Columns additionally reported in metrics | 3 engineered features |
| Target column | None detected |
| String columns | 1 |
| Binary integer features | 15 |
| Multi-valued integer feature | `legs` |
| Engineered features | 3 |
| Missing values | None reported |

The primary metadata profile describes 17 columns:

- `animal_name`
- 15 biological or behavioral attributes
- `legs`

The metrics artifact additionally reports:

- `terrestrial_mammalian_score`
- `environmental_adaptation_score`
- `locomotion_structure_score`

These additional fields appear to have been generated during analysis rather than included in the original 17-column profile. Their presence should be reconciled before downstream modeling or productionization.

### 2.2 Entity identifier assessment

`animal_name` is a string field with cardinality 100 across 101 rows. The value `frog` appears twice, while the other displayed examples appear once:

- `frog`: 2 occurrences
- `aardvark`: 1 occurrence
- `antelope`: 1 occurrence

Therefore, `animal_name` is not fully unique and should not automatically be treated as a primary key. It should generally be excluded from predictive modeling unless animal identity is explicitly meaningful and repeated entities are handled appropriately.

---

## 3. Data Quality Assessment

### 3.1 Missing values

No missing values were reported in the metadata profile or metrics artifact. Every profiled column has:

- `missing_count = 0`
- `missing_pct = 0.0`

No imputation was performed. The configured imputation rules were:

- Median imputation for numeric columns with absolute skewness greater than 1.
- Mean imputation for numeric columns with absolute skewness less than or equal to 1.
- Mode imputation for categorical or string columns.
- No imputation when no values are missing.

Because the dataset contains no reported missing values, these rules had no operational effect on the analyzed data.

### 3.2 Schema and artifact consistency

A material data-quality issue is the difference between the reported dimensions and the listed metrics columns:

- The dataset overview reports 101 rows and 17 columns.
- The metrics column summary includes 17 original/profiled columns plus 3 engineered features.
- The engineered features have no associated `profile_type` or `profile_key_metric` in the summary.

Recommended remediation:

1. Confirm whether the three engineered features are physically present in the source file or were created during the pipeline.
2. Maintain separate metadata for source columns and derived columns.
3. Recompute the final post-feature-engineering schema and dimensions.
4. Ensure visualizations, correlation matrices, and modeling inputs reference the same data version.

### 3.3 Outlier interpretation

The outlier analysis used the IQR rule. For many binary variables, the first and third quartiles are identical, yielding an IQR of zero. As a result, the less common binary value is labeled as an outlier. These are generally not numeric anomalies; they represent minority biological states.

Examples include:

| Feature | Reported outliers | Percentage | Interpretation |
|---|---:|---:|---|
| `airborne` | 24 | 23.76% | Minority value in a binary trait |
| `feathers` | 20 | 19.80% | Minority value in a binary trait |
| `breathes` | 21 | 20.79% | Minority value in a binary trait |
| `backbone` | 18 | 17.82% | Minority value in a binary trait |
| `fins` | 17 | 16.83% | Minority value in a binary trait |
| `domestic` | 13 | 12.87% | Minority value in a binary trait |
| `venomous` | 8 | 7.92% | Minority value in a binary trait |
| `legs` | 2 | 1.98% | Potentially unusual leg counts under IQR rules |

The apparent outliers in binary fields should not be removed without domain justification. For `legs`, the two observations beyond the IQR upper bound of 7 may merit a direct record-level review, although the values may be biologically valid.

---

## 4. Feature Distribution Findings

### 4.1 Binary attribute prevalence

Most biological attributes are binary and encoded as 0 or 1. For these fields, the mean approximates the proportion of observations with value 1.

| Feature | Mean | Approximate prevalence of value 1 | Median | Skewness note |
|---|---:|---:|---:|---|
| `hair` | 0.43 | 43% | 0 | No high-skew flag |
| `feathers` | 0.20 | 20% | 0 | High positive skew: 1.54 |
| `eggs` | 0.58 | 58% | 1 | No high-skew flag |
| `milk` | 0.41 | 41% | 0 | No high-skew flag |
| `airborne` | 0.24 | 24% | 0 | High positive skew: 1.25 |
| `aquatic` | 0.36 | 36% | 0 | No high-skew flag |
| `predator` | 0.55 | 55% | 1 | No high-skew flag |
| `toothed` | 0.60 | 60% | 1 | No high-skew flag |
| `backbone` | 0.82 | 82% | 1 | High negative skew: -1.71 |
| `breathes` | 0.79 | 79% | 1 | High negative skew: -1.46 |
| `venomous` | 0.08 | 8% | 0 | High positive skew: 3.16 |
| `fins` | 0.17 | 17% | 0 | High positive skew: 1.80 |
| `tail` | 0.74 | 74% | 1 | High negative skew: -1.13 |
| `domestic` | 0.13 | 13% | 0 | High positive skew: 2.25 |
| `catsize` | 0.44 | 44% | 0 | No high-skew flag |

The most prevalent positive attributes are:

1. `backbone`: 82%
2. `breathes`: 79%
3. `tail`: 74%
4. `toothed`: 60%
5. `eggs`: 58%

The least prevalent positive attributes are:

1. `venomous`: 8%
2. `domestic`: 13%
3. `fins`: 17%
4. `feathers`: 20%
5. `airborne`: 24%

The combination of a small sample size and rare traits such as `venomous`, `domestic`, and `fins` creates a risk of unstable estimates and unstable model coefficients.

### 4.2 `legs`

| Statistic | Value |
|---|---:|
| Range | 0 to 8 |
| Cardinality | 6 |
| Mean | 2.84 |
| Median | 4 |
| Q1 | 2 |
| Q3 | 4 |
| IQR | 2 |
| IQR lower bound | -1 |
| IQR upper bound | 7 |
| Outlier count | 2 |

`legs` is discrete rather than continuous, with six distinct values. Its median exceeds its mean, indicating that lower leg counts pull the average downward. It should be treated as an ordinal or discrete biological feature rather than assumed to be a normally distributed continuous variable.

---

## 5. Correlation Analysis

Because there is no target variable, the correlations are descriptive associations among predictors and engineered features. They should not be interpreted as causal relationships.

### 5.1 Strongest positive correlations

| Feature 1 | Feature 2 | Correlation |
|---|---|---:|
| `milk` | `terrestrial_mammalian_score` | 0.915 |
| `locomotion_structure_score` | `tail` | 0.903 |
| `hair` | `milk` | 0.879 |
| `hair` | `terrestrial_mammalian_score` | 0.866 |
| `backbone` | `locomotion_structure_score` | 0.851 |
| `aquatic` | `environmental_adaptation_score` | 0.804 |
| `backbone` | `tail` | 0.732 |
| `environmental_adaptation_score` | `fins` | 0.709 |
| `breathes` | `terrestrial_mammalian_score` | 0.670 |
| `airborne` | `feathers` | 0.657 |

The strongest positive relationships are concentrated in two interpretable groups:

- Mammalian or terrestrial structure:
  - `hair`
  - `milk`
  - `backbone`
  - `breathes`
  - `terrestrial_mammalian_score`
- Locomotion and structural support:
  - `backbone`
  - `tail`
  - `locomotion_structure_score`

### 5.2 Strongest negative correlations

| Feature 1 | Feature 2 | Correlation |
|---|---|---:|
| `eggs` | `milk` | -0.939 |
| `eggs` | `terrestrial_mammalian_score` | -0.846 |
| `eggs` | `hair` | -0.817 |
| `eggs` | `toothed` | -0.642 |
| `aquatic` | `breathes` | -0.638 |
| `breathes` | `fins` | -0.617 |
| `feathers` | `toothed` | -0.614 |
| `fins` | `legs` | -0.606 |
| `airborne` | `toothed` | -0.594 |
| `environmental_adaptation_score` | `hair` | -0.530 |

The strong negative relationship between `eggs` and `milk` is biologically interpretable within this dataset and indicates substantial separation between egg-laying and milk-producing animals. However, the magnitude may also reflect the small, structured sample and should be validated if used for inference.

### 5.3 Multicollinearity implications

Several feature pairs are sufficiently correlated to create redundancy concerns, especially for linear models:

- `eggs` and `milk`
- `hair` and `milk`
- `milk` and `terrestrial_mammalian_score`
- `hair` and `terrestrial_mammalian_score`
- `backbone` and `locomotion_structure_score`
- `tail` and `locomotion_structure_score`
- `aquatic` and `environmental_adaptation_score`

For tree-based models, correlated features are less problematic for prediction but can make feature importance difficult to interpret because importance may be distributed across interchangeable variables. For linear models, regularization and coefficient stability analysis are recommended.

---

## 6. Feature Engineering Highlights

Three engineered features were reported.

### 6.1 Engineered feature definitions

| Feature | Formula or method | Type | Intended purpose |
|---|---|---|---|
| `terrestrial_mammalian_score` | `hair + milk + backbone + breathes` | int64 | Summarizes traits associated with mammalian and terrestrial biological structure |
| `environmental_adaptation_score` | `aquatic + fins + airborne` | int64 | Captures adaptation to aquatic and aerial environments |
| `locomotion_structure_score` | Normalized `legs + tail + backbone` | float64 | Represents structural traits related to movement and body support |

### 6.2 Engineered feature distributions

| Feature | Cardinality | Outlier count | Outlier percentage |
|---|---:|---:|---:|
| `terrestrial_mammalian_score` | 5 | 0 | 0.00% |
| `environmental_adaptation_score` | 3 | 0 | 0.00% |
| `locomotion_structure_score` | 10 | 0 | 0.00% |

The engineered features are strongly correlated with their source attributes, as expected:

- `terrestrial_mammalian_score` with `milk`: 0.915
- `locomotion_structure_score` with `tail`: 0.903
- `environmental_adaptation_score` with `aquatic`: 0.804
- `environmental_adaptation_score` with `fins`: 0.709

These scores may improve interpretability by aggregating related biological traits. However, they also introduce deterministic redundancy. Their use should be evaluated through cross-validation rather than assumed to improve generalization.

---

## 7. Statistical Hypothesis Testing

No feature-level hypothesis tests were conducted because no explicit target column was detected.

| Assessment | Result |
|---|---|
| Target detected | No |
| Target correlations | None |
| Significant predictors | None |
| Significance threshold | 0.05 |
| Statistical inference | Not available |

The absence of a target means that supervised significance testing, such as comparing feature distributions across outcome classes, cannot be performed. A class label or outcome variable must be identified before calculating predictor significance, target associations, or classification performance.

---

## 8. Visualization Artifact Descriptions

### 8.1 `correlation_matrix.png`

- Artifact type: Image visualization
- File size: 739.46 KB
- Purpose: Visual representation of pairwise correlations among original and engineered numeric features
- Main analytical themes reflected in the metrics:
  - Strong positive association between mammalian traits and `terrestrial_mammalian_score`
  - Strong positive association between structural traits and `locomotion_structure_score`
  - Strong negative association between `eggs` and `milk`
  - Positive association between environmental traits and `environmental_adaptation_score`

The numerical correlation values in `metrics.json` should be treated as the authoritative summary for exact interpretation.

### 8.2 `target_interactions.png`

- Artifact type: Image visualization
- File size: 35.7 KB
- Purpose: Intended to show feature interactions with a target
- Limitation: No target column was detected, and no target interaction metrics are available
- Interpretation: The artifact should not be used as evidence of supervised target relationships until the target definition and plotting logic are confirmed

The presence of this file alongside a null target configuration warrants an artifact validation check.

---

## 9. Predictive Modeling Blueprint

### 9.1 Target definition

The current dataset has no explicit target. Modeling should not proceed as a supervised learning exercise until a target is confirmed, such as:

- A predefined animal class
- A taxonomic category
- An ecological or behavioral class
- Another domain-specific outcome

The proposed problem type remains undefined until this label is supplied.

### 9.2 Recommended algorithms

Once a target is available, the following sequence is appropriate for this small, predominantly binary dataset:

1. **Regularized logistic regression**
   - Interpretable baseline
   - Useful for assessing linear separability
   - Requires attention to multicollinearity and scaling of continuous engineered features

2. **Constrained decision tree**
   - Provides interpretable biological rules
   - Use shallow depth and minimum leaf constraints to reduce overfitting

3. **Random forest**
   - Captures nonlinear relationships and interactions
   - Consider class weighting if the target is imbalanced

4. **Gradient boosting or XGBoost-style boosting**
   - Potentially effective for nonlinear interactions
   - Requires strong regularization and careful validation due to the small sample

5. **Linear or kernel SVM**
   - Suitable for small datasets
   - Requires scaling for continuous features and careful hyperparameter tuning

### 9.3 Feature handling

Recommended feature-selection and preprocessing practices include:

- Exclude `animal_name` unless entity identity is explicitly relevant.
- Include engineered scores only after checking whether they add out-of-sample value.
- Use cross-validated permutation importance and mutual information for feature ranking.
- Avoid selecting features using the full dataset before cross-validation.
- Treat binary indicators consistently as numeric or categorical variables.
- Scale continuous engineered features for logistic regression and SVM models.
- Do not scale features solely for tree-based models.
- Use one-hot encoding if additional categorical predictors are introduced.

### 9.4 Validation strategy

Given the sample size of 101 observations:

- Use stratified k-fold cross-validation for classification.
- Prefer repeated stratified cross-validation to quantify variability.
- Use nested cross-validation when tuning hyperparameters.
- Reserve a final holdout set only if sufficient observations remain after training and validation.
- Report:
  - Balanced accuracy
  - Macro F1
  - Per-class recall
  - Confusion matrix
  - ROC-AUC where applicable
  - Confidence intervals or fold-level variability where feasible

The dataset size is too small to rely on a single train-test split as the primary performance estimate.

### 9.5 Overfitting controls

Recommended safeguards are:

- Shallow decision trees
- Minimum leaf-size constraints
- Regularization for linear models
- Early stopping for boosting algorithms
- Class weighting where outcome imbalance exists
- Avoidance of high-cardinality name encoding
- Monitoring of train-validation performance gaps
- Stability checks for feature importance across folds
- Preference for simpler models when performance differences are not robust

---

## 10. Recommended Next Steps

### Immediate data and artifact validation

1. Confirm the intended source schema and reconcile the 17-column profile with the 20-column metrics summary.
2. Determine whether engineered features are source fields or pipeline-generated fields.
3. Verify that `target_interactions.png` is valid despite the absence of a target.
4. Review the two `legs` observations flagged by the IQR procedure.
5. Confirm the duplicate `frog` record and determine whether it represents repeated data or a naming issue.

### Analytical next steps

1. Define and document the supervised target, if predictive modeling is required.
2. Confirm whether the dataset is intended for classification, clustering, descriptive analysis, or rule discovery.
3. Examine distributions and co-occurrence patterns for rare attributes such as `venomous`, `domestic`, and `fins`.
4. Compare models with and without engineered features using repeated cross-validation.
5. Assess multicollinearity and coefficient stability for linear models.
6. Use domain review to distinguish valid biological minority patterns from data errors.

---

## 11. Executive Conclusion

This EDA identifies a compact, complete, highly structured animal-attribute dataset with predominantly binary biological variables. The strongest relationships are biologically coherent and center on mammalian traits, environmental adaptation, and locomotion structure. The engineered features provide interpretable summaries but are inherently redundant with their component variables.

The main limitation is not data completeness but analytical specification: no target variable is present. Consequently, target-based inference, predictor significance, and supervised model evaluation are not currently available. Before production modeling, the target definition, final schema, and visualization consistency must be resolved. Once those issues are addressed, the recommended approach is to establish a regularized interpretable baseline and compare it with constrained tree ensembles using repeated, leakage-safe cross-validation.