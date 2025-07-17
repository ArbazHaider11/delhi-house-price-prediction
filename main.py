import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import r2_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

#creating a pipeline
def build_pipeline(num_att, cat_att):
    pip_num = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler())])
    
    pip_cat = Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore"))
                        ])

    
    full_pipeline = ColumnTransformer([("num", pip_num, num_att),
                                       ("cat", pip_cat, cat_att)])
    
    return full_pipeline
                    

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("housing.csv")

    df = df[df["Parking"] != 39] #removing the outlier row

    df["Price_lakhs_cat"] = pd.cut(df["Price_lakhs"], bins = [0,50,100,150,np.inf], labels = [1,2,3,4])
    split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
    for train_index, test_index in split.split(df, df["Price_lakhs_cat"]):
        train = df.iloc[train_index].drop("Price_lakhs_cat", axis = 1)
        test = df.iloc[test_index].drop("Price_lakhs_cat", axis = 1).to_csv("input.csv", index=False)

    train_copy = train.copy()

    #creatijg features and labels
    housing_features = train_copy.drop("Price_lakhs", axis=1)
    housing_labels = train_copy["Price_lakhs"]

    #splitting the data
    housing_num = housing_features.select_dtypes(include=["float64", "int64"]).columns.tolist()
    housing_cat = housing_features.select_dtypes(include=["object"]).columns.tolist()

    #fitting inside the pipeline
    pipeline = build_pipeline(housing_num, housing_cat)
    housing_prepared = pipeline.fit_transform(housing_features)

    #trainig the model
    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    # After model.fit(...) and before joblib.dump(...)
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=housing_labels, y=model.predict(housing_prepared))
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.plot([housing_labels.min(), housing_labels.max()],
            [housing_labels.min(), housing_labels.max()],
            color='red', lw=2)
    plt.show()

    #creating files
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    cv_score = -cross_val_score(model, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    r2 = r2_score(housing_labels, model.predict(housing_prepared))
    print(f"R² Score on training data: {r2:.2f}")
    print("")
    print(f"The cross value score of the model is {cv_score}")

    

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input_data = pipeline.transform(input_data)
    prediction = model.predict(transformed_input_data)

    input_data["Predicted_price_lakhs"] = prediction

    input_data.to_csv("output.csv", index=False)
    print("Predictions saved to output.csv file")
    