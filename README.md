# Delhi House Price Prediction 🏠

This project implements a complete machine learning pipeline to predict house prices in Delhi using **scikit-learn** and a **Random Forest Regressor**. It includes steps from **data cleaning** and **feature engineering** to **model training**, **evaluation**, and **saving outputs**.

---

## 📌 Project Overview

The goal of this project is to create a price prediction model for residential properties in Delhi based on features like area, number of bedrooms/bathrooms, furnishing status, locality, parking availability, etc.

Although implemented on a **small-scale dataset**, the pipeline is **scalable** and can easily accommodate **larger datasets** with similar structure and go through the **same preprocessing, modeling, and prediction flow**.

---

## 🛠️ Tools & Technologies

- Python
- JupyterLab (for data analysis & cleaning)
- Visual Studio Code (for core development & model training)
- scikit-learn
- pandas, NumPy
- Seaborn, Matplotlib
- joblib

---

## 🔍 Workflow

1. **Data Cleaning** (performed in JupyterLab):
   - Handled missing values.
   - Removed outliers (e.g., unrealistic parking values).
   - Stratified sampling based on price category to ensure fair train-test split.

2. **Feature Engineering**:
   - Created numerical and categorical pipelines using `ColumnTransformer`.
   - Handled scaling, imputation, and one-hot encoding.

3. **Model Training** (done in VSCode):
   - Built and trained a `RandomForestRegressor` on the processed dataset.
   - Performed 10-fold cross-validation.
   - Evaluated performance using R² score and RMSE.

4. **Visualization**:
   - Plotted Actual vs Predicted prices to inspect model fit.

5. **Deployment Readiness**:
   -Entire pipeline saved using joblib, allowing seamless deployment and reusability.
   -The input test set is automatically transformed and passed through the model.
   - Input test set is automatically transformed and predicted.
   - Predictions are saved to `output.csv`.

7. **Model Comparison

To determine the most suitable algorithm for this regression task, multiple models were trained and evaluated using 10-fold cross-validation based on Root Mean Squared Error (RMSE). The comparison is as follows:

| Model              | Mean RMSE |
|-------------------|-----------|
| Linear Regression | 163.80    |
| Decision Tree     | 148.47    |
| Random Forest     | 107.14    |

✅ **Random Forest** was chosen as the final model due to its significantly better performance in terms of prediction error.

---

## 📂 Project Structure
housing.csv # Dataset
├── input.csv # Test data after stratified split
├── output.csv # Predictions (generated)
├── main.py # Main logic for training or predicting
├── model.pkl # Trained Random Forest model (joblib)
├── pipeline.pkl # Feature pipeline (joblib)
├── README.md # Project documentation


---

## 📈 Future Improvements

- Add a user-friendly Flask-based web interface (currently in progress).
- Hyperparameter tuning and ensemble experimentation.
- More granular handling of categorical features like locality.
- Expand support for real-world deployment (e.g., Streamlit, Docker).

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and build upon it with attribution.

---

## 🙋‍♂️ Author

**Arbaz Haider**  
Third-year B.Tech CSE (AIML Specialization)  
[LinkedIn](https://www.linkedin.com/in/arbazhaider11042005/)

---

## 🔗 GitHub Repository

[👉 View the Project on GitHub](https://github.com/your-username/delhi-house-price-prediction)

---

*This is a demonstration-level project. However, the architecture and modular design make it easily extendable to production-scale datasets and use cases.*


