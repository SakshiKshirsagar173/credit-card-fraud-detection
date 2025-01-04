# credit-card-fraud-detection
This Python script implements a machine learning pipeline for detecting fraudulent credit card transactions using a Random Forest Classifier. Below is a step-by-step breakdown of the code:

1. **Import Libraries**:
   - `numpy` and `pandas` for data manipulation and numerical computations.
   - `matplotlib` and `seaborn` for data visualization, specifically for plotting a confusion matrix.
   - `sklearn` for machine learning tools, including preprocessing, model building, and evaluation.

2. **Load and Inspect Data**:
   - The dataset (`creditcard.csv`) is loaded into a Pandas DataFrame.
   - `df.head()` is called to display the first few rows of the dataset.
   - The script checks for missing values using `df.isnull().sum()` to ensure data quality.

3. **Data Preprocessing**:
   - The `Amount` column is normalized using `StandardScaler` to bring the values within a similar range and improve the model’s performance.
   - The `Time` column is dropped from the dataset as it is assumed not to contribute significantly to the classification task.

4. **Feature Selection**:
   - The target variable `Class` is separated from the feature set `x`, and the dataset is split into training and testing sets (80% training, 20% testing) using `train_test_split`.

5. **Model Training**:
   - A Random Forest Classifier is initialized with 100 estimators (trees) and trained on the training data (`x_train`, `y_train`).

6. **Model Prediction & Evaluation**:
   - The trained model makes predictions on the test set (`x_test`), and the accuracy score is calculated.
   - A classification report is printed, showing precision, recall, and F1-score for both classes (fraudulent and non-fraudulent transactions).
   - A confusion matrix is computed and visualized using a heatmap to show the model’s performance in terms of true positives, true negatives, false positives, and false negatives.

7. **Results**:
   - The model achieves an impressive accuracy of **99.96%**.
   - The classification report shows high precision for class 0 (non-fraudulent transactions), but lower recall for class 1 (fraudulent transactions), indicating that while the model is excellent at identifying non-fraudulent transactions, it struggles more with detecting fraud.
   - The confusion matrix confirms this, with a large number of true negatives (non-fraudulent transactions correctly classified) and fewer true positives (fraudulent transactions correctly classified).

### Key Observations:
- **Accuracy**: The model performs exceptionally well in terms of overall accuracy, but given the class imbalance (the dataset is heavily skewed towards non-fraudulent transactions), the model’s ability to detect fraudulent transactions (class 1) could be improved.
- **Class Imbalance**: The high accuracy is largely driven by the model's ability to correctly classify the majority class (non-fraudulent). The recall for the minority class (fraudulent transactions) is lower, which suggests that additional techniques such as oversampling, undersampling, or using more advanced algorithms might improve performance in detecting fraud.

### Visual:
- The confusion matrix heatmap provides a clear view of the model's performance in distinguishing between the two classes.

This pipeline is useful for fraud detection tasks, where class imbalance is a common challenge, and improving the model's recall for detecting fraud is crucial.
