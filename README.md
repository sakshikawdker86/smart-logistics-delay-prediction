# Smart Logistics Delay Prediction 🚚

## Project Overview
This project analyzes e-commerce delivery data and builds a machine learning classification model to predict whether an order is likely to be delivered late.

The goal is to identify potential delivery delays and understand the factors associated with late deliveries, helping logistics teams take proactive measures and improve customer experience.

## Business Problem
Late deliveries negatively impact customer satisfaction and trust. Predicting potential delays in advance can help businesses prioritize shipments, improve logistics planning, and reduce delivery-related issues.

## Objectives
- Analyze historical e-commerce delivery data.
- Identify patterns associated with late deliveries.
- Perform data cleaning and feature engineering.
- Build classification models to predict delivery delays.
- Handle imbalanced target classes.
- Evaluate models using Precision, Recall, and F1-score.
- Explore decision threshold tuning to improve late-delivery detection.

## Dataset
**Olist Brazilian E-Commerce Dataset**

The dataset contains e-commerce order information, delivery timestamps, customer details, seller details, and order-item information.

### Target Variable
| Value | Description |
|---|---|
| `0` | On-time delivery |
| `1` | Late delivery |

Orders were classified as late when the actual delivery date was later than the estimated delivery date.

## Project Workflow
1. Data loading and preprocessing
2. Exploratory Data Analysis (EDA)
3. Outlier investigation
4. Feature engineering
5. Train-validation-test split
6. Model training
7. Class imbalance handling
8. Threshold tuning
9. Final model evaluation

## Exploratory Data Analysis
The EDA explored:

- On-time vs. late delivery distribution
- Actual and estimated delivery duration
- Delivery delay distributions and outliers
- Geographical delivery patterns across customer and seller states
- Order price and freight distributions
- Order item counts
- Feature correlations and relationships with late deliveries

### Key Findings
- Approximately **6.77%** of orders were late.
- The dataset is highly imbalanced.
- Most orders were delivered earlier than their estimated dates.
- Delivery delays varied across geographical regions.
- Price and freight distributions were right-skewed.
- Selected numerical features showed weak individual linear correlations with late-delivery outcomes.

## Feature Engineering
Order-item data was aggregated into order-level features.

| Feature | Description |
|---|---|
| `estimated_delivery_days` | Estimated delivery duration |
| `total_price` | Total product price per order |
| `total_freight` | Total freight charges per order |
| `item_count` | Number of items in an order |

### Target
`is_late` was created using actual and estimated delivery dates.

Features derived from actual delivery outcomes, such as actual delivery duration and delivery delay, were excluded from the predictors to avoid target leakage.

## Models Evaluated
- Logistic Regression — baseline model
- Balanced Logistic Regression — class-weighted model
- Random Forest Classifier — tree-based model

`StandardScaler` was used for Logistic Regression. Random Forest was trained without scaling.

## Class Imbalance
Late deliveries accounted for approximately 6.77% of the dataset.

To improve minority-class detection, class balancing was applied using:

`class_weight='balanced'`

Accuracy was not treated as the only evaluation metric because a model could achieve high accuracy while failing to detect late deliveries.

## Threshold Tuning
Prediction thresholds were evaluated using the validation dataset.

Precision, Recall, and F1-score were compared at different thresholds to understand the trade-off between identifying late deliveries and generating false-positive alerts.

The selected threshold was **0.50**, based on the highest F1-score among the tested thresholds for Balanced Logistic Regression.

## Final Test Results

**Selected Model:** Balanced Logistic Regression  
**Selected Threshold:** 0.50

| Metric | Score |
|---|---:|
| Accuracy | 58.90% |
| Late Precision | 8.36% |
| Late Recall | 50.88% |
| Late F1-score | 14.36% |

### Confusion Matrix

| | Predicted On-time | Predicted Late |
|---|---:|---:|
| Actual On-time | 10,699 | 7,288 |
| Actual Late | 642 | 665 |

### Interpretation
The model identified 665 late deliveries out of 1,307 actual late deliveries, achieving approximately 50.88% recall.

However, precision was low at 8.36%, indicating that many orders predicted as late were actually on time. This highlights the limitations of the selected features and the difficulty of predicting late deliveries in an imbalanced dataset.

## Limitations
- Low precision and a high number of false-positive predictions.
- Limited predictive power from the selected numerical features.
- High class imbalance.
- Limited geographical and logistics-related predictive information.
- Weak individual correlations between numerical predictors and the target.

## Future Improvements
- Include customer and seller geographical features.
- Incorporate product categories and product characteristics.
- Explore historical seller delivery performance.
- Add richer logistics and order timing features.
- Experiment with additional feature engineering and classification models.
- Further optimize the precision-recall trade-off using business requirements.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Project Structure (Suggested)

```text
Smart-Logistics-Delay-Prediction/
│
├── data/
│   └── raw/
│
├── notebooks/
│   └── smart_logistics_delay_prediction.ipynb
│
├── README.md
└── requirements.txt
```

## Conclusion
This project demonstrates an end-to-end machine learning workflow, from data preprocessing and exploratory analysis to feature engineering, model training, class balancing, threshold tuning, and final evaluation.

The results show that detecting late deliveries is challenging with the selected features. While Balanced Logistic Regression identified approximately half of the late orders, its low precision indicates that further feature engineering and richer logistics information are required for more reliable predictions.

The project also highlights the importance of choosing appropriate evaluation metrics for imbalanced classification problems and keeping the final test set separate from model selection.