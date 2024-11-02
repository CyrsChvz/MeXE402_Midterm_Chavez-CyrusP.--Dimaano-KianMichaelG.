<a name="readme-top"> </a>
# MeXE402_Midterm_Chavez-CyrusP.--Dimaano-KianMichaelG.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li> <a href="#Introduction">Introduction</a> </li>
    <li> <a href="#Dataset Description">Dataset Description</a> </li>
    <li> <a href="#Project Objectives">Project Objectives</a> </li>
    <li> <a href="#Linear Regression Analysis">Linear Regression Analysis</a> </li>
    <li> <a href="#Logistic Regression Analysis">Logistic Regression Analysis</a> </li>
    <li> <a href="#Documentation">Documentaion</a> </li>
  
  </ol>
</details>

<a name="Introduction"> </a>
# I. Introduction
<div align="justify">
  
Linear and logistic regression are foundational techniques in statistical modeling and machine learning, often used for predicting outcomes based on input variables. Despite their similarities, they are tailored to solve different types of problems: linear regression is used for predicting continuous outcomes, while logistic regression is used for predicting categorical outcomes.

### Linear Regression
Linear regression models the relationship between a dependent variable (output) and one or more independent variables (inputs). It aims to find the best-fit line through the data points that minimizes the difference between the actual and predicted values.

- **Formula**: The linear regression model is represented as:
> ![formula 1](https://github.com/user-attachments/assets/13c0ee67-dd84-4da8-9cdb-7f91460825fa)

  where:
  - ![Capture](https://github.com/user-attachments/assets/838efc3d-188a-4f4d-9719-5e9701e926b0) is the dependent variable,
  - ![2](https://github.com/user-attachments/assets/83e757c2-f4d1-47e8-8331-6cbc12e8b928) are independent variables,
  - ![3](https://github.com/user-attachments/assets/37130a27-443b-4bfa-b638-c55b638a01f9) is the intercept,
  - ![4](https://github.com/user-attachments/assets/36a968e7-f66e-4eec-896d-268536bf5144) are the coefficients
  - ![5](https://github.com/user-attachments/assets/b3143738-1bec-4482-9743-d797ee35cf4e) represents the error term.
- **Goal**: Minimize the sum of squared errors (differences between actual and predicted values).
- **Interpretability**: Each coefficient (beta) reflects the estimated change in \( y \) for a one-unit change in the corresponding \( x \), holding other variables constant.
- **Use Case**: Predicting continuous outcomes, such as predicting house prices, stock prices, or temperatures.

### Logistic Regression
Logistic regression, on the other hand, is used for binary classification tasks, where the output variable is categorical. Instead of predicting an exact value, it estimates the probability that a given instance falls into a particular category.

- **Formula**: Logistic regression uses the logistic (sigmoid) function to map linear combinations of inputs to a probability range between 0 and 1:
 > ![6](https://github.com/user-attachments/assets/29f1d4b4-4511-46f5-a453-08688fc133e4)
 > ##### where \( P(y=1|X) \) is the probability of the positive class (e.g., \( y=1 \)).
- **Goal**: Maximize the likelihood of correctly classifying the data points.
- **Interpretability**: Coefficients indicate the log-odds of the outcome associated with each predictor.
- **Use Case**: Binary or multiclass classification problems, such as spam detection, disease prediction, or customer churn prediction.

### Key Differences
1. **Type of Output**:
   - Linear regression outputs continuous values.
   - Logistic regression outputs probabilities (typically mapped to binary outcomes).

2. **Loss Function**:
   - Linear regression uses Mean Squared Error (MSE).
   - Logistic regression uses Log Loss (or Cross-Entropy Loss) to optimize.

3. **Interpretation**:
   - In linear regression, coefficients directly reflect the change in the target variable.
   - In logistic regression, coefficients reflect changes in log-odds for the target variable.

### Summary
- **Linear Regression**: Best for predicting continuous outcomes by minimizing error between predictions and actual values.
- **Logistic Regression**: Best for binary (or categorical) outcomes, focusing on classifying data points by estimating probabilities. 

Both techniques are widely used, easy to interpret, and form the basis for more complex machine learning models.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="Dataset Description"> </a>
# II. Dataset Description
<div align="justify">

> ## The data set used for linear regression:
#### The California Housing Prices dataset on Kaggle includes information about housing in various California districts. Key features includes: 
- **median house value** 
- **median income** 
- **average house age** 
- **total rooms** 
- **total bedrooms** 
- **population** 
- **households**
#### in each block group. This dataset is widely used for regression tasks, particularly for predicting housing prices based on demographic and geographic information.

#### In order to achieve the optimal performance of data processing for Linear Regression, we replaced some of the data with numerical values including:

- **ocean proximity categorized as:**
  > #### (1) - Near ocean
  > #### (2) - Near bay
  > #### (3) - <1h ocean
  > #### (4) - Inland
  

> ## The data set used for logistic regression:
#### The Kaggle customer churn dataset contains attributes related to customer demographics, account information, and service usage. Key features includes:

- **customer tenure** 
- **contract type** 
- **payment method** 
- **monthly charges** 
- **total charges** 
- **target variable, "Churn"** which indicates if a customer has left service 

#### It’s designed for predictive modeling, primarily binary classification, to analyze factors influencing customer churn.

#### In order to achieve the optimal performance of data processing for Logistic Regression, we replaced some of the data with numerical values including:

- **Male - 0**
- **Female - 1**
- **Basic - 1**
- **Standard - 2**
- **Premium - 3**
- **Monthly - 1**
- **Quarterly - 3**
- **Annually - 12**
- **Churned - 1**
- **Not Churned - 0**

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="Project Objectives"> </a>
# III. Project Objectives
<div align="justify">

> ### 1. **California Housing Prices Dataset**:  
   Build a Linear regression model to predict **housing prices** across California districts. Using features like median income, house age, and population, the project aims to assess key drivers of housing prices and provide a tool for price estimation that supports real estate decision-making and market analysis. 

> ### 2. **Customer Churn Dataset**:  
   Develop a predictive model using Logistic regression to identify customers at high risk of **churning**, enabling proactive engagement strategies. By analyzing customer demographics, service usage, and payment methods, the goal is to uncover key factors influencing churn and suggest actionable insights to improve customer retention.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="Linear Regression Analysis"> </a>
# IV. Linear Regression Analysis
<div align="justify">

> ### Methodology for California Housing Prices Prediction using Linear Regression

### Step 1: Import Libraries and Load Dataset
The dataset is loaded using `pandas`. The file path is set for an Excel file named `housing.xlsx`.

```python
import pandas as pd
dataset = pd.read_excel('housing.xlsx')
```

### Step 2: Inspect the Dataset
Display the first five rows of the dataset to understand its structure.

```python
dataset.head(5)
```

### Step 3: Define Features (X) and Target Variable (y)
Separate the features and target variable. Here, all columns except the last are used as features (`X`), while the last column is set as the target (`y`).

```python
X = dataset.iloc[:, :-1].values
print(X)  # Display features

y = dataset.iloc[:, -1].values
print(y)  # Display target variable
```

### Step 4: Split the Dataset into Training and Test Sets
Split the data with an 80-20 split for training and testing, using a fixed random state for reproducibility.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Step 5: Initialize and Train the Model
Here, a `HistGradientBoostingRegressor` is used, which is suitable for larger datasets and can handle missing values better than typical linear regression.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)
```

### Step 6: Predict on the Test Set
Use the trained model to make predictions on the test set and print these predictions.

```python
y_pred = model.predict(X_test)
print(y_pred)  # Display predictions
```

Additionally, a specific prediction is made with given input values.

```python
print(model.predict([[-122.23, 37.88, 41, 880, 129.0, 322, 126, 8.3252, 2]]))
```

### Step 7: Evaluate Model Performance with R-squared
The R-squared metric is calculated to assess the model’s performance on the test set.

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)
```

### Step 8: Calculate Adjusted R-squared
Calculate the Adjusted R-squared to account for the number of predictors.

```python
k = X_test.shape[1]  # Number of features
n = X_test.shape[0]  # Number of observations

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
adj_r2  # Display adjusted R-squared
```
### Conclusion

This methodology uses a gradient boosting model to predict housing prices. The process includes loading data, splitting it into training and testing sets, and training a model to make predictions. Model accuracy is evaluated using R-squared and Adjusted R-squared, showing how well the predictions match actual prices. This approach provides a straightforward way to analyze and predict housing prices based on various features.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="Logistic Regression Analysis"> </a>
# V. Logistic Regression Analysis
<div align="justify">

> ### Methodology for Customer Churn Prediction using Logistic Regression

#### Step 1: **Data Loading and Initial Exploration**

1. **Import Libraries**:
   - Import `pandas` for data handling, `scikit-learn` for model training and evaluation, and visualization libraries (`matplotlib` and `seaborn`).
   
2. **Load the Dataset**:
   ```python
   import pandas as pd
   dataset = pd.read_csv('customer_churn_dataset-testing-master.csv')
   ```
   - Load the dataset, which contains information on customer churn, into a DataFrame.

3. **Preview the Dataset**:
   ```python
   dataset.head(10)
   ```
   - Display the first 10 rows of the dataset to understand its structure and identify relevant features.

---

#### Step 2: **Data Preparation**

1. **Feature Selection and Target Variable**:
   - Separate features (`X`) from the target variable (`y`) for model training:
     ```python
     X = dataset.iloc[:, 1:-1].values  # Select all columns except the first and last
     y = dataset.iloc[:, -1].values    # Target column is assumed to be the last
     ```

2. **Splitting Data into Training and Test Sets**:
   - Split the data into training and testing subsets with an 80/20 split:
     ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
     ```

3. **Feature Scaling**:
   - Apply standard scaling to ensure features are on a similar scale, a common step before logistic regression:
     ```python
     from sklearn.preprocessing import StandardScaler
     sc = StandardScaler()
     X_train = sc.fit_transform(X_train)
     X_test = sc.transform(X_test)  # Use the same scaler for the test set
     ```

---

#### Step 3: **Model Training with Logistic Regression**

1. **Initialize and Train the Logistic Regression Model**:
   - Instantiate the `LogisticRegression` model and fit it to the training data:
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression(random_state=0)
     model.fit(X_train, y_train)
     ```

---

#### Step 4: **Model Evaluation**

1. **Generate Predictions**:
   - Use the trained model to predict customer churn on the test set:
     ```python
     y_pred = model.predict(X_test)
     ```

2. **Confusion Matrix**:
   - Calculate and display the confusion matrix to evaluate the model’s performance:
     ```python
     from sklearn.metrics import confusion_matrix
     conf_matrix = confusion_matrix(y_test, y_pred)
     print(conf_matrix)
     ```

   - **Visualize the Confusion Matrix**:
     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns

     plt.figure(figsize=(6, 4))
     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
     plt.title('Confusion Matrix')
     plt.ylabel('Actual Values')
     plt.xlabel('Predicted Values')
     plt.show()
     ```

3. **Accuracy Score**:
   - Calculate the model’s accuracy on the test set:
     ```python
     from sklearn.metrics import accuracy_score
     accuracy = accuracy_score(y_test, y_pred)
     print(accuracy)
     ```

4. **Individual Prediction (Optional)**:
   - Use the model to predict churn for a single data point:
     ```python
     print(model.predict(sc.transform([[22, 0, 25, 14, 4, 27, 1, 1, 598, 9]])))
     ```

---

### Conclusion

This approach provides a structured pipeline for data preparation, model training, and performance evaluation, focusing on accuracy and the confusion matrix. Adjustments or additional steps could include further feature engineering or optimizing the model based on insights from the initial evaluation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<a name="Documentation"> </a>
# VI. Documentation

## Code Comments

> ## Linear Regression

```python
import pandas as pd
dataset = pd.read_excel('housing.xlsx')
```

```python
dataset.head(5)
```

```python
X = dataset.iloc[:, :-1].values
print(X)  # Display features

y = dataset.iloc[:, -1].values
print(y)  # Display target variable
```

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

```python
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor(random_state=0)
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
print(y_pred)  # Display predictions
```

```python
print(model.predict([[-122.23, 37.88, 41, 880, 129.0, 322, 126, 8.3252, 2]]))
```

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)
```

```python
k = X_test.shape[1]  # Number of features
n = X_test.shape[0]  # Number of observations

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
adj_r2  # Display adjusted R-squared
```
---

> ## Logistic Regression

   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns

   dataset = pd.read_csv('customer_churn_dataset-testing-master.csv')
   ```
   
   ```python
   dataset.head(10)
   ```

  ```python
     X = dataset.iloc[:, 1:-1].values  # Select all columns except the first and last
     y = dataset.iloc[:, -1].values    # Target column is assumed to be the last
  ```

  ```python
     from sklearn.model_selection import train_test_split
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  ```

  ```python
     from sklearn.preprocessing import StandardScaler
     sc = StandardScaler()
     X_train = sc.fit_transform(X_train)
     X_test = sc.transform(X_test)  # Use the same scaler for the test set
  ```

  ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression(random_state=0)
     model.fit(X_train, y_train)
  ```

  ```python
     y_pred = model.predict(X_test)
  ```

  ```python
     from sklearn.metrics import confusion_matrix
     conf_matrix = confusion_matrix(y_test, y_pred)
     print(conf_matrix)
  ```

  ```python
     import matplotlib.pyplot as plt
     import seaborn as sns

     plt.figure(figsize=(6, 4))
     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
     plt.title('Confusion Matrix')
     plt.ylabel('Actual Values')
     plt.xlabel('Predicted Values')
     plt.show()
  ```

  ```python
     from sklearn.metrics import accuracy_score
     accuracy = accuracy_score(y_test, y_pred)
     print(accuracy)
  ```

  ```python
     print(model.predict(sc.transform([[22, 0, 25, 14, 4, 27, 1, 1, 598, 9]])))
  ```
---

## Results and Discussion

### Linear Regression
The Independent Variables of California Housing Prices Dataset are:
- Median House Value
- Median Income
- Average House Age
- Total Rooms
- Total Bedrooms
- Population
- Households
  
On the other hand, the Dependent Variable is the variable **Housing Prices**.

Since some of the variables have unique values, it has to be replaced with numerical values.

Below is a **Scatter Plot** of the **Housing Median Value** showing the result of Linear Regression with 83-83 percent accuracy.

![scatter plot](https://github.com/user-attachments/assets/96e5dfab-456c-41d3-98a7-4b396aabd0da)

---

### Discussion

This project aimed to construct a linear regression model to predict housing prices across various districts in California. Using features such as median income, house age, and population, the analysis sheds light on the primary drivers of housing prices and presents a practical model for real estate decision-making and market analysis.

#### Key Insights:
1. **Importance of Median Income**: From the analysis, it is evident that median income plays a pivotal role in determining housing prices. The model highlights a strong positive correlation between median income and housing prices, suggesting that wealthier districts tend to have higher property values. This finding underscores the economic disparities between regions and reflects the impact of income levels on real estate markets.

2. **Influence of House Age and Population**: While median income is the most influential factor, other features like house age and population also contribute to price variations, albeit to a lesser extent. House age may reflect the desirability of established neighborhoods or the quality of older constructions, while population density could influence housing demand and pricing dynamics. However, these variables showed less pronounced effects compared to income.

3. **Model Performance**: The linear regression model demonstrates reasonable predictive capabilities, as evidenced by the R-squared value and other performance metrics. While the model captures significant trends in the data, there may be room for improvement, especially in cases where non-linear relationships or interactions between features are present. Further enhancements, such as using more complex models (e.g., polynomial regression or ensemble methods), could be explored to improve accuracy.

4. **Practical Implications for Real Estate**: The model provides a foundational tool for estimating housing prices based on easily accessible demographic and geographical data. Real estate professionals and market analysts can use these predictions to inform pricing strategies, investment decisions, and market evaluations. By understanding the drivers of housing prices, stakeholders can make more informed choices in a dynamic market.

5. **Limitations and Future Work**: Despite the model's utility, it has limitations. Linear regression assumes a linear relationship between features and housing prices, which may not capture more complex patterns in the data. Additionally, external factors such as economic trends, policy changes, and geographical nuances are not accounted for. Future work could involve incorporating more features, experimenting with non-linear models, or conducting a more granular analysis of specific California regions.

#### Conclusion
The linear regression model provides valuable insights into the determinants of housing prices, with median income emerging as a key predictor. Although the model serves as a useful starting point for price estimation, it highlights the complexity of real estate markets and the need for continuous refinement. By leveraging such models, real estate stakeholders can enhance their decision-making processes, but they should remain aware of the model's assumptions and potential limitations.

--- 


### Logistic Regression
The Independent Variables of Customer Churn Dataset are:
- Customer Tenure
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
  
On the other hand, the Dependent Variable is the variable **Churn**.

Since some of the variables have unique values, it has to be replaced with numerical values.

Below is a **Heat Map** of the **Confusion Matrix** showing the result of Logistic Regression with 82-82 percent accuracy.

![image](https://github.com/user-attachments/assets/3821ef4f-1f47-4c2a-8689-876808483e2c)

---

### Discussion

An insight from the logistic regression model reveals that customers with shorter tenure and those on month-to-month contracts have a significantly higher risk of churning compared to customers on annual contracts or those with longer tenure. This is often observed through the model’s coefficients, where short tenure and monthly contract variables would have higher positive coefficients, indicating a strong association with churn.

This suggests that offering incentives, such as discounts for longer-term commitments or loyalty rewards for customers who renew annually, could be effective in reducing churn. Additionally, providing targeted support or personalized engagement for newer customers can help strengthen their relationship with the company, potentially decreasing their likelihood of leaving.

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Linear vs. Logistic Regression: Key Differences and Limitations

> **Purpose and Output**:
   - **Linear Regression**: Used for predicting continuous outcomes (e.g., House pricing in California) by assuming a linear relationship between features and target values.
   - **Logistic Regression**: Designed for binary classification (e.g., churn prediction). It models probabilities between 0 and 1 using a sigmoid function.

> **Interpretation**:
   - **Linear Regression**: Coefficients indicate the change in the outcome for each feature.
   - **Logistic Regression**: Coefficients indicate the change in log odds, helping interpret how features impact the probability of churn.

> **Limitations**:
   - **Linear Regression**: Cannot handle binary outcomes and is sensitive to outliers; it assumes linear relationships, limiting it in cases with complex data patterns.
   - **Logistic Regression**: Assumes linearity in log odds, struggles with non-linear relationships, and can be challenging to interpret with many variables. It also performs poorly on imbalanced datasets unless adjustments are made.

Logistic regression’s focus on binary outcomes and interpretability make it ideal for churn prediction, though limitations include sensitivity to data balance and linear assumptions. Each regression has their own strengths if used accordingly.

<div align="justify">

<p align="right">(<a href="#readme-top">back to top</a>)</p>
### References

> ### https://www.kaggle.com/datasets/camnugent/california-housing-prices/code
> ### https://www.kaggle.com/datasets/blastchar/telco-customer-churn/code
