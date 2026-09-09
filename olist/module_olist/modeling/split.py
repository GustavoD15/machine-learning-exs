import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "purchase_hour",
    "promised_days",
    "purchase_weekday",
    "purchase_month",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight",
    "customer_state"
]

TARGET = "is_late"

def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the input DataFrame into training and testing sets.
    """
    X = data[FEATURES]
    y = data[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Ensure the split maintains the proportion of classes in the target variable
    )
    return X_train, X_test, y_train, y_test