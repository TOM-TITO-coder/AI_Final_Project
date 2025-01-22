import streamlit as st
import pandas as pd
import joblib
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PowerTransformer
import numpy as np

class ChangeDataType(BaseEstimator, TransformerMixin):
    # Converts specified columns to datetime format.
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = pd.to_datetime(X[col], errors='coerce')
        return X

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    # Extracts date and time-related features like hour, month, day of the week, and part of the day.
    def __init__(self, date_column, transaction_hour_bins, transaction_hour_labels):
        self.date_column = date_column
        self.transaction_hour_bins = transaction_hour_bins
        self.transaction_hour_labels = transaction_hour_labels
        self.new_columns = ['transaction_hour', 'transaction_month', 'is_weekend', 'day_of_week', 'part_of_day']
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['transaction_hour'] = X[self.date_column].dt.hour
        X['transaction_month'] = X[self.date_column].dt.month
        X['is_weekend'] = X[self.date_column].dt.weekday.isin([5, 6]).astype(int)
        
        # Day of week: Monday=0, Sunday=6
        X['day_of_week'] = X[self.date_column].dt.day_name()
        
        # Part of day classification
        X['part_of_day'] = pd.cut(X['transaction_hour'], 
                                  bins=self.transaction_hour_bins, 
                                  labels=self.transaction_hour_labels, 
                                  right=True)
        return X

class AgeFeature(BaseEstimator, TransformerMixin):
    # Calculates age based on the date of birth (DOB) column.
    def __init__(self, dob_column):
        self.dob_column = dob_column
        self.new_column = 'age'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        reference_date = pd.Timestamp(2020, 12, 31)
        X[self.new_column] = (reference_date - X[self.dob_column]).dt.days // 365
        return X


class CalculateDistance(BaseEstimator, TransformerMixin):
    # Calculates the distance between two geographical points using the Haversine formula.
    def __init__(self, lat_col, long_col, merch_lat_col, merch_long_col):
        self.lat_col = lat_col
        self.long_col = long_col
        self.merch_lat_col = merch_lat_col
        self.merch_long_col = merch_long_col
        self.new_column = 'distance'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # Convert latitudes and longitudes to radians
        lat1 = np.radians(X[self.lat_col])
        lon1 = np.radians(X[self.long_col])
        lat2 = np.radians(X[self.merch_lat_col])
        lon2 = np.radians(X[self.merch_long_col])
        
        # Haversine formula to calculate distance
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371  # Radius of the Earth in kilometers
        X[self.new_column] = R * c  # Distance in kilometers
        
        return X

class BinCityPopulation(BaseEstimator, TransformerMixin):
    # Groups city population into bins with specified labels.
    def __init__(self, city_pop_bins, city_pop_labels):
        self.city_pop_bins = city_pop_bins
        self.city_pop_labels = city_pop_labels
        self.new_column = 'city_pop_bin'

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_column] = pd.cut(X['city_pop'], bins=self.city_pop_bins, labels=self.city_pop_labels)
        return X
    

class YeoJohnsonTransformer(BaseEstimator, TransformerMixin):
    # Applies the Yeo-Johnson transformation to normalize the 'amt' column.
    def __init__(self):
        self.transformer = PowerTransformer(method='yeo-johnson')
        self.new_column = 'amt_yeo_johnson'

    def fit(self, X, y=None):
        self.transformer.fit(X[['amt']])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.new_column] = self.transformer.transform(X[['amt']])
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    # Drops specified columns from the dataset.
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=self.columns, errors='ignore')
        self.remaining_columns = X.columns
        return X


class LabelEncoding(BaseEstimator, TransformerMixin):
    # Performs label encoding for specified categorical columns.
    def __init__(self, columns):
        self.columns = columns
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            le = LabelEncoder()
            le.fit(X[col])
            self.label_encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col] = self.label_encoders[col].transform(X[col])
        return X

class ScaleFeatures(BaseEstimator, TransformerMixin):
    # Scales numerical features to a range of 0 to 1 using MinMaxScaler.
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = X.copy()
        X[:] = self.scaler.transform(X)
        return X

class CFraud(nn.Module):
    def __init__(self, layers_sz, in_sz, out_sz):
        super(CFraud, self).__init__()
        layers = []
        for sz in layers_sz:
            layers.append(nn.Linear(in_sz, sz))
            in_sz = sz
        self.linears = nn.ModuleList(layers)
        self.out = nn.Linear(layers_sz[-1], out_sz)
        self.act_func = nn.ReLU()
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.linears:
            x = self.act_func(layer(x))
        x = self.output_activation(self.out(x))
        return x

def preprocess_input(input_df):
    from sklearn.pipeline import Pipeline

    # Preprocessing pipeline
    city_pop_bins = [0, 10000, 50000, 100000, 500000, 1000000, np.inf]
    city_pop_labels = ['<10K', '10K-50K', '50K-100K', '100K-500K', '500K-1M', '>1M']

    transaction_hour_bins=[-1, 5, 11, 17, 21, 24]
    transaction_hour_labels=['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']

    drop_columns = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'amt', 
                    'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 
                    'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']

    categorical_features = ['category', 'gender', 'day_of_week', 'part_of_day', 'city_pop_bin']

    preprocessor = Pipeline([
        ('change_dtype', ChangeDataType(columns=['trans_date_trans_time', 'dob'])),
        ('datetime_features', DateTimeFeatures(date_column='trans_date_trans_time',
                                            transaction_hour_bins = transaction_hour_bins,
                                            transaction_hour_labels = transaction_hour_labels)),
        ('age_feature', AgeFeature(dob_column='dob')),
        ('calculate_distance', CalculateDistance(lat_col='lat', long_col='long', 
                                                merch_lat_col='merch_lat', merch_long_col='merch_long')),
        ('bin_city_pop', BinCityPopulation(city_pop_bins = city_pop_bins, city_pop_labels = city_pop_labels)),
        ('yeo_johnson', YeoJohnsonTransformer()),
        ('drop_columns', DropColumns(columns=drop_columns)),
        ('label_encoding', LabelEncoding(columns=categorical_features)),
        ('scale_features', ScaleFeatures()),
    ])
    
    try:
        # Ensure no missing values in critical columns before processing
        input_df = input_df.fillna(0)

        # Apply the preprocessing pipeline
        processed_data = preprocessor.fit_transform(input_df)

        # Ensure the output is a NumPy array
        if isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data.to_numpy()

        return processed_data

    except Exception as e:
        raise ValueError(f"Error during preprocessing: {e}")
    

def load_model(model_path):
    return joblib.load(model_path)

def main():
    st.title("Credit Card Fraud Detection App (CFraud APP)")
    st.write("Enter transaction details to predict whether it's fraudulent.")

    # Sidebar for model selection
    st.sidebar.title("Select Model")
    model_choice = st.sidebar.radio(
        "Choose a model to predict:",
        ("Model 1: Logistic Regression", "Model 2: Random Forest", "Model 3: Artificial Neural Network")
    )

    # File uploader for CSV input
    st.sidebar.markdown("[Example CSV input file](#)")
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file:
        input_data = pd.read_csv(uploaded_file)
        st.subheader("Input Data Preview")
        st.write(input_data)
    else:
        st.write("Please upload a CSV file.")

    # Load the selected model
    if model_choice == "Model 1: Logistic Regression":
        model = load_model("../models/lg_model1.pt")
    elif model_choice == "Model 2: Random Forest":
        model = load_model("../models/rf_model1.pt")
    elif model_choice == "Model 3: Artificial Neural Network":
        # model = CFraud(layers_sz=[300, 150], in_sz=11, out_sz=1)
        model = torch.load("../models/ann_model_update_10.pt")
        model.eval()

    if st.button("Predict") and uploaded_file:
        # Preprocess the input data
        try:
            preprocessed_data = preprocess_input(input_data)

            if model_choice == "Model 3: Artificial Neural Network":
                # Ensure data is converted to PyTorch tensor
                preprocessed_data = torch.tensor(preprocessed_data, dtype=torch.float32)

                # Perform prediction
                with torch.no_grad():
                    outputs = model(preprocessed_data)
                    outputs = outputs.squeeze()

                    # Ensure outputs are iterable
                    if outputs.ndimension() == 0:  # Single prediction case
                        outputs = outputs.unsqueeze(0)

                    fraud_probs = torch.sigmoid(outputs).squeeze().tolist()
                    predictions = [1 if prob > 0.5 else 0 for prob in fraud_probs]

            else:
                # Predictions for other models
                predictions = model.predict(preprocessed_data)
                fraud_probs = model.predict_proba(preprocessed_data)[:, 1]

            # Show results
            results = pd.DataFrame({
                "Prediction": ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions],
                "Fraud Probability": fraud_probs
            })
            st.write("Prediction Results")
            st.write(results)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Run the app
if __name__ == "__main__":
    main()
