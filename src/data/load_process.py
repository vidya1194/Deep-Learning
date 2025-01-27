import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.utils.utils import setup_logging

logger = setup_logging()

def load_data(file_path):
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        logger.error(f"Error in load_data: {e}")
        raise e


def process_data(df):
    try:
        
        # Separating target variable and other variables
        Y = df.Attrition
        X = df.drop(columns=['Attrition'])

        # Scaling the data
        sc = StandardScaler()
        X_scaled = sc.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Splitting the data
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=1, stratify=Y)

        return x_train, x_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error in process_data: {e}")
        raise e
    