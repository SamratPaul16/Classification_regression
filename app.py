import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

# Initialize session state
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = {}

def load_data():
    st.sidebar.title("Load Data")
    data_option = st.sidebar.selectbox("Choose a dataset", ["Iris", "Tips", "Titanic", "Diamonds", "Flights", "Upload File"])
    
    if data_option == "Upload File":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "tsv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.tsv'):
                df = pd.read_csv(uploaded_file, sep='\t')
            st.session_state.df = df
    else:
        try:
            df = sns.load_dataset(data_option.lower())
            st.session_state.df = df
        except ValueError:
            st.error(f"Dataset '{data_option}' not found. Please choose a different dataset.")

def preprocess_data(df):
    st.title("Data Preprocessing")
    st.write("Column Names:", df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Data Head:")
    st.write(df.head())
    st.write("Data Shape:", df.shape)
    st.write("Data Describe:")
    st.write(df.describe())
    st.write("Null Values:")
    st.write(df.isnull().sum())

    selected_column = st.selectbox('Select hue column', df.columns)
    # Create a pair plot
    st.subheader('Pair plot')
    st.pyplot(sns.pairplot(df, hue=selected_column))

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Impute numerical columns with mean
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    # Impute categorical columns with most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Encode categorical variables
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    st.session_state.encoders = encoders

    return df

def select_features_target(df):
    st.title("Select Features and Target")
    target = st.selectbox("Select the target variable", df.columns.tolist())
    features = st.multiselect("Select the feature variables", df.columns.tolist(), default=[col for col in df.columns if col != target])
    X = df[features]
    y = df[target]
    st.session_state.target = target
    return X, y, target

def determine_problem_type(y):
    if y.dtype in ['int64', 'float64']:
        # Check the target is continuous or discrete
        if len(y.unique()) > 10:
            return 'Regression'
        else:
            return 'Classification'
    elif y.dtype in ['object', 'category']:
        return 'Classification'
    else:
        raise ValueError("Unsupported target data type")

def select_model(problem_type):
    st.title("Select the Model")
    if problem_type == 'Regression':
        model_options = ["Linear Regression", "Support Vector Regression (SVR)", 
                         "Decision Tree Regression", "Random Forest Regression", "Gradient Boosting Regression"]
    else:
        model_options = ["Support Vector Machines (SVM)", "Decision Trees Classifier", "Random Forest Classifier", 
                         "Gradient Boosting Machines (GBM)", "K-Nearest Neighbors (KNN)"]
    model_name = st.selectbox(f"Select the model for {problem_type}", model_options)
    return model_name

def get_model_params(model_name, problem_type):
    st.title("Model Parameters")
    params = {}
    if model_name in ["Random Forest Regression", "Decision Tree Regression", "Random Forest Classifier", "Decision Trees Classifier"]:
        params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=100, value=50)
        params['max_depth'] = st.slider("Max Depth", min_value=3, max_value=20, value=10)
    if model_name == "Gradient Boosting Regression" or model_name == "Gradient Boosting Machines (GBM)":
        params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=100, value=50)
        params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.2, value=0.1)
    if model_name == "K-Nearest Neighbors (KNN)":
        params['n_neighbors'] = st.slider("Number of Neighbors", min_value=1, max_value=15, value=5)
    params['test_size'] = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)
    return params

def train_model(X, y, model_name, problem_type, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=42)

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Support Vector Regression (SVR)":
        model = SVR()
    elif model_name == "Decision Tree Regression":
        model = DecisionTreeRegressor(max_depth=params['max_depth'])
    elif model_name == "Random Forest Regression":
        model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    elif model_name == "Gradient Boosting Regression":
        model = GradientBoostingRegressor(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    elif model_name == "Support Vector Machines (SVM)":
        model = SVC(probability=True)
    elif model_name == "Decision Trees Classifier":
        model = DecisionTreeClassifier(max_depth=params['max_depth'])
    elif model_name == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    elif model_name == "Gradient Boosting Machines (GBM)":
        model = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate=params['learning_rate'])
    elif model_name == "K-Nearest Neighbors (KNN)":
        model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, problem_type)
    st.session_state.model = model
    st.session_state.metrics = metrics
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

   # Save the trained model to the Downloads folder
    save_model(model)
    return metrics, model, X_test, y_test

def save_model(model):
    model_filename = "trained_model.pkl"
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_path = os.path.join(models_dir, model_filename)
    with open(model_path, 'wb') as file:
        joblib.dump(model, file)
    st.success(f"Model saved successfully at {model_path}")
    # Provide a download link
    with open(model_path, 'rb') as f:
        st.download_button(
            label="Download Model",
            data=f,
            file_name=model_filename,
            mime="application/octet-stream"
        )
def evaluate_model(y_test, y_pred, problem_type):
    if problem_type == 'Regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()
           # Get original class labels
        if st.session_state.target in st.session_state.encoders:
            classes = st.session_state.encoders[st.session_state.target].classes_
        else:
            classes = np.unique(y_test)
        
        return {
            'Accuracy': accuracy, 
            'Precision': precision, 
            'Recall': recall, 
            'F1 Score': f1, 
            'Confusion Matrix': cm, 
            'Classification Report': cr_df, 
            'Classes': classes
        }
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)


def make_prediction(input_data, model=None):
    if model is None:
        model = st.session_state.model
    if model is None:
        st.error("No model available for prediction. Please train a model first.")
        return
    input_df = pd.DataFrame([input_data])
    # Convert input data to appropriate types
    for col in input_df.columns:
        if input_df[col].dtype in ['int64', 'float64']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            input_df[col].fillna(0, inplace=True)
        else:
            # Encode categorical variables using the same encoders from training
            if col in st.session_state.encoders:
                input_df[col] = st.session_state.encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)
     # Assuming target was encoded, decode the prediction
    if isinstance(st.session_state.encoders.get(st.session_state.target), LabelEncoder):
        prediction = st.session_state.encoders[st.session_state.target].inverse_transform(prediction)
    
    st.write(f"Predicted {st.session_state.target}: {prediction[0]}")

def main():
    st.title("Machine Learning App")
    st.write("Welcome to the Machine Learning App! This app helps you with regression and classification tasks.")

    load_data()
    if 'df' in st.session_state:
        df = st.session_state.df
        df = preprocess_data(df)
        X, y, target = select_features_target(df)
        problem_type = determine_problem_type(y)
        model_name = select_model(problem_type)
        params = get_model_params(model_name, problem_type)

        # Train Model
        if st.button("Train Model"):
            metrics, model, X_test, y_test = train_model(X, y, model_name, problem_type, params)
            # Download Model
       

        # Make Prediction
        input_data = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                input_data[col] = st.text_input(f"Enter value for {col}", key=f"input_{col}")
            else:
                input_data[col] = st.selectbox(f"Select value for {col}", df[col].unique(), key=f"input_{col}")
        st.session_state.input_data = input_data  # Store input data in session state

        if st.button("Predict"):
            make_prediction(input_data, st.session_state.model)

        # Display stored input data and model in sidebar
        st.sidebar.title("Session State")
        if 'input_data' in st.session_state:
            st.sidebar.write("Input Data:")
            st.sidebar.write(st.session_state.input_data)
        else:
            st.sidebar.write("Input Data: No input data available.")
        if'model' in st.session_state:
            st.sidebar.write("Model Name: " + st.session_state.model.__class__.__name__)
        else:
            st.sidebar.write("Trained Model: No model available for evaluation.")

        if'metrics' in st.session_state:
            st.write("Evaluation Metrics:")
            for key, value in st.session_state.metrics.items():
                if key == 'Confusion Matrix':
                   classes = st.session_state.metrics['Classes']
                   plot_confusion_matrix(value, classes=classes)
                elif key == 'Classification Report':
                   st.dataframe(value)
                else:
                    st.write(f"{key}: {value}")

if __name__ == "__main__":
    main()
