# Used Car Price Prediction Using Machine Learning

**CodeAlpha Data Science Internship — Task 3**  
**Author:** Vishnu Harikaran  
**Dataset:** CarDekho Vehicle Dataset (`car data.csv`)

---

## 1. Project Title
**Used Car Price Prediction Using Supervised Machine Learning and Scikit-Learn Pipelines**

---

## 2. Abstract
Accurately estimating the resale price of a pre-owned vehicle is critical for buyers, sellers, dealerships, and automotive financing platforms. This project develops an end-to-end machine learning regression pipeline on the CarDekho Vehicle Dataset (301 raw listings, 299 unique records after deduplication) to predict used vehicle selling prices (`Selling_Price`, in Lakhs INR). After exploratory data analysis, duplicate removal, domain validation, and feature engineering (`Car_Age` derived from manufacturing `Year`), five regression models—Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and Extra Trees Regressor—were trained and evaluated using 5-fold cross-validation and an 80:20 stratified holdout test split. **Gradient Boosting Regressor** achieved the strongest test-set performance (**R² = 0.9072**, **RMSE = 1.1803 Lakhs INR**, **MAE = 0.6215 Lakhs INR**), closely followed by **Extra Trees Regressor** (**R² = 0.9036**, **RMSE = 1.2029 Lakhs INR**, **MAE = 0.5946 Lakhs INR**). The complete fitted preprocessing and prediction pipeline is saved to `models/best_car_price_model.joblib` for direct reuse without retraining.

---

## 3. Introduction
Used vehicle marketplaces require transparent, data-driven pricing tools because vehicle resale value depends simultaneously on original showroom price, vehicle age, mileage, fuel type, transmission, seller channel, and ownership history. Rule-of-thumb depreciation tables fail to capture non-linear interactions across these attributes. Machine learning regression models—especially ensemble tree methods—learn complex depreciation patterns directly from historical transaction listings while maintaining reproducible preprocessing through `scikit-learn` pipelines.

---

## 4. Problem Statement
Given a vehicle's characteristics—manufacturing year, current showroom price (`Present_Price`), kilometres driven (`Kms_Driven`), fuel type (`Fuel_Type`), seller type (`Seller_Type`), transmission type (`Transmission`), and previous owner count (`Owner`)—predict its market selling price (`Selling_Price`, in Lakhs INR) accurately and without data leakage.

---

## 5. Objectives
1. Perform comprehensive data exploration and quality checks (missing values, duplicates, data types, target skewness, and outlier inspection).
2. Clean and preprocess the dataset inside leak-free `scikit-learn` pipelines (`StandardScaler` for numeric features, `OneHotEncoder` for categorical features).
3. Engineer meaningful features (`Car_Age`) and remove redundant or noisy high-cardinality fields (`Year`, `Car_Name`).
4. Visualize feature relationships and distributions across 10 EDA charts and 5 model evaluation charts.
5. Train, cross-validate, and compare five regression algorithms using MAE, MSE, RMSE, and R² Score.
6. Persist the best fitted pipeline and metadata using Joblib and demonstrate standalone inference on unseen car specifications.

---

## 6. Dataset Description
- **Source:** [Kaggle – Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (`car data.csv`)
- **Raw Shape:** 301 rows × 9 columns
- **Cleaned Shape:** 299 rows × 9 columns (after removing 2 exact duplicate rows)
- **Target Variable:** `Selling_Price` (float64, range `0.10` to `35.00` Lakhs INR, mean `4.66`, median `3.60`, skewness `2.4934`)

| Column | Type | Description | Range / Categories |
|---|---|---|---|
| `Car_Name` | String | Vehicle model name | 98 unique names (cars & two-wheelers) |
| `Year` | Integer | Manufacturing year | 2003 – 2018 |
| `Selling_Price` | Float | Target resale price (Lakhs INR) | 0.10 – 35.00 |
| `Present_Price` | Float | Current showroom price (Lakhs INR) | 0.32 – 92.60 |
| `Kms_Driven` | Integer | Total distance driven (km) | 500 – 500,000 |
| `Fuel_Type` | Categorical | Fuel category | Petrol (239), Diesel (58*), CNG (2) |
| `Seller_Type` | Categorical | Listing channel | Dealer (193*), Individual (106) |
| `Transmission` | Categorical | Gearbox type | Manual (260*), Automatic (39*) |
| `Owner` | Integer | Previous owners count | 0, 1, or 3 |

*\*Counts reflect the 299 deduplicated records.*

---

## 7. Data Preprocessing
1. **Whitespace & Type Normalization:** Stripped whitespace from string columns and verified integer/float schemas.
2. **Missing Values:** Verified `0` missing values across all 9 columns.
3. **Duplicate Removal:** Identified and removed **2 exact duplicate records** (one 2016 `ertiga` entry and one 2015 `fortuner` entry), yielding **299** unique rows.
4. **Outlier Investigation:** Upper IQR fence checks flagged high-end SUVs (`land cruiser` at `Present_Price = 92.60`, `Selling_Price = 35.00`; `fortuner` at `33.00`) and high-mileage vehicles (`500,000` km `Activa 3g`). Every record satisfied `Selling_Price < Present_Price` and valid physical bounds, so legitimate high-value/high-mileage observations were retained rather than blindly discarded.
5. **Leakage Prevention:** Train-test splitting (80:20, `239` train / `60` test, stratified across `Selling_Price` quintiles, `random_state=42`) was performed before fitting `StandardScaler` and `OneHotEncoder(handle_unknown="ignore")` inside `ColumnTransformer`. `Selling_Price` was strictly isolated as the target `y`.

---

## 8. Exploratory Data Analysis
Ten visualizations were generated and saved in `outputs/visualizations/`:
- **Target Distribution (`01_selling_price_distribution.png`):** Right-skewed distribution (`mean = 4.66`, `median = 3.60`), reflecting a mix of budget two-wheelers, mid-size hatchbacks/sedans, and premium SUVs.
- **Present Price vs Selling Price (`02_present_price_vs_selling_price.png`):** Strongest linear relationship (`r = 0.8764`), confirming original showroom price anchors resale value.
- **Car Age vs Selling Price (`03_car_age_vs_selling_price.png`):** Negative correlation (`r = -0.2344`); older vehicles depreciate steadily.
- **Kilometres Driven vs Selling Price (`04_kms_driven_vs_selling_price.png`):** Higher mileage correlates with older vehicles (`r = 0.5257` with `Car_Age`) and lower resale value within comparable vehicle tiers.
- **Categorical Drivers (`05`–`07`, `09`):** Diesel vehicles average `10.10` Lakhs vs Petrol (`3.26`) and CNG (`3.10`); Automatic vehicles average `9.07` Lakhs vs Manual (`3.92`); Dealer listings average `6.63` Lakhs vs Individual listings (`0.87`).
- **Correlation Heatmap (`08_correlation_heatmap.png`):** Highlights `Present_Price` (`r = 0.8764`) and `Car_Age` (`r = -0.2344`) as key numeric signals.
- **Tree Feature Importance (`10_feature_importance.png`):** Confirms `Present_Price` (`0.8803`) and `Car_Age` (`0.0764`) dominate predictive importance.

---

## 9. Feature Engineering
- **`Car_Age`:** Derived deterministically as `2026 - Year` (stored as `reference_year = 2026` in model metadata to guarantee identical calculation during training and inference).
- **Column Pruning:** Dropped `Year` after deriving `Car_Age` (redundant collinear feature) and dropped `Car_Name` (98 unique labels across 299 rows would cause sparse high-cardinality overfitting, while `Present_Price` already encodes vehicle segment).

---

## 10. Machine Learning Algorithms
All five required regressors were wrapped in identical preprocessing `Pipeline` objects:
1. **Linear Regression:** Ordinary least squares baseline modeling linear relationships over scaled numeric and one-hot encoded features.
2. **Decision Tree Regressor:** Non-linear recursive partitioning baseline (`random_state=42`).
3. **Random Forest Regressor:** Bagged ensemble of 200 decision trees (`n_estimators=200`, `random_state=42`) reducing single-tree variance.
4. **Gradient Boosting Regressor:** Sequential additive boosting of regression trees (`random_state=42`) iteratively minimizing squared-error residuals.
5. **Extra Trees Regressor:** Randomized ensemble of 200 trees (`n_estimators=200`, `random_state=42`) using random split thresholds.

---

## 11. Model Evaluation Metrics
Models were evaluated via 5-fold cross-validation (`KFold(n_splits=5, shuffle=True, random_state=42)`) on the 239 training samples and on the 60 holdout test samples using:
- **MAE (Mean Absolute Error):** Average absolute error in Lakhs INR.
- **MSE (Mean Squared Error):** Mean squared prediction error, penalizing large misses.
- **RMSE (Root Mean Squared Error):** Square root of MSE in Lakhs INR.
- **R² Score (Coefficient of Determination):** Proportion of target variance explained by the model.

---

## 12. Model Comparison
Actual metrics recorded in `outputs/model_comparison.csv`:

| Model | CV R² Mean | CV R² Std | CV RMSE Mean | Train R² | Test MAE | Test MSE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting Regressor** | **0.8590** | **0.1065** | **1.7775** | **0.9970** | **0.6215** | **1.3932** | **1.1803** | **0.9072** |
| Extra Trees Regressor | 0.8629 | 0.0554 | 1.7972 | 1.0000 | 0.5946 | 1.4470 | 1.2029 | 0.9036 |
| Linear Regression | 0.7798 | 0.0801 | 2.2839 | 0.8828 | 1.1965 | 3.1208 | 1.7666 | 0.7921 |
| Random Forest Regressor | 0.8611 | 0.1110 | 1.7861 | 0.9890 | 0.8382 | 3.7620 | 1.9396 | 0.7494 |
| Decision Tree Regressor | 0.8700 | 0.0359 | 1.7833 | 1.0000 | 0.8728 | 4.6664 | 2.1602 | 0.6891 |

---

## 13. Best Model Selection
**Gradient Boosting Regressor** was selected as the final production model because it achieved the highest test-set **R² Score (`0.9072`)** and lowest **MSE (`1.3932`)** and **RMSE (`1.1803` Lakhs INR)**, with strong 5-fold cross-validation consistency (`CV R² = 0.8590`, `CV RMSE = 1.7775`). While Extra Trees Regressor achieved a slightly lower MAE (`0.5946` vs `0.6215`) and slightly higher CV R² (`0.8629`), Gradient Boosting produced lower squared-error variance on the test set and avoided memorizing the training set (`Train R² = 0.9970` vs `1.0000`).

---

## 14. Prediction Demonstration
Using the saved bundle `models/best_car_price_model.joblib` via `src/predict.py` (no retraining):

| Sample Vehicle | Year | Present Price | Kms Driven | Fuel | Seller | Transmission | Owner | Derived `Car_Age` | Predicted `Selling_Price` |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| `ciaz` | 2017 | 9.85 | 6,900 | Petrol | Dealer | Manual | 0 | 9 yrs | **7.85 Lakhs INR** |
| `fortuner` | 2015 | 30.61 | 40,000 | Diesel | Dealer | Automatic | 0 | 11 yrs | **22.94 Lakhs INR** |
| `Royal Enfield Classic 350` | 2016 | 1.47 | 12,000 | Petrol | Individual | Manual | 0 | 10 yrs | **1.17 Lakhs INR** |

---

## 15. Results and Discussion
- **Dominant Drivers:** Feature importance from the selected Gradient Boosting pipeline confirms `Present_Price` (`0.8803`) and `Car_Age` (`0.0764`) drive over 95% of predictive splits, followed by `Kms_Driven` (`0.0199`) and `Fuel_Type_Diesel` (`0.0168`).
- **Residual Behavior:** Evaluation plots (`outputs/model_evaluation/`) show predicted prices align closely with the `y = x` ideal line and residuals center tightly around zero.
- **Notable Error Case:** In `outputs/prediction_results.csv`, the largest individual test residual (`-6.6721` Lakhs INR) occurs on a 2006 `camry` (`Present_Price = 23.73`, `Kms_Driven = 142,000`, `Owner = 3`, actual `2.50` vs predicted `9.1721`), because `Owner = 3` appears only once in the entire dataset.

---

## 16. Limitations
1. **Small Dataset Size (299 unique records):** Rare segments (`CNG` with 2 records, `Owner = 3` with 1 record, ultra-luxury SUVs) have limited training support.
2. **Mixed Vehicle Classes:** The CarDekho `car data.csv` file includes both passenger cars and two-wheelers without an explicit vehicle-type column, though `Present_Price` separates them well.
3. **Tree Extrapolation Bounds:** Tree-based ensembles cannot extrapolate resale prices beyond the maximum target range seen during training.
4. **Static Market Snapshot:** Listings span manufacturing years 2003–2018; real-world deployment on 2026+ vehicle prices would require retraining on current market data.

---

## 17. Conclusion
This project delivers a complete, reproducible used-car price prediction system built with modular Python source code, an executable Jupyter notebook, 15 diagnostic and EDA visualizations, and a serialized `scikit-learn` pipeline. **Gradient Boosting Regressor** reliably explains **90.72%** of test-set variance (`RMSE = 1.1803` Lakhs INR, `MAE = 0.6215` Lakhs INR) and provides instant, validated price predictions via `src/predict.py`.

---

## 18. References
1. Birla, N. *Vehicle Dataset from CarDekho* (`car data.csv`). Kaggle. https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
2. Pedregosa et al. *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825–2830, 2011.
3. Hunter, J. D. *Matplotlib: A 2D Graphics Environment*, Computing in Science & Engineering, 2007.
4. Waskom, M. *Seaborn: Statistical Data Visualization*, Journal of Open Source Software, 2021.
