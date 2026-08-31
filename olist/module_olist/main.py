from xgboost import data
from module_olist.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import load_data, save_data
from module_olist.features import create_dataset, create_features
from module_olist.modeling.split import split_data

ORDERS_PATH = RAW_DATA_DIR / "olist_orders_dataset.csv"
ITEMS_PATH = RAW_DATA_DIR / "olist_order_items_dataset.csv"
PRODUCTS_PATH = RAW_DATA_DIR / "olist_products_dataset.csv"  # Alterado de CUSTOMERS para PRODUCTS
CUSTOMERS_PATH = RAW_DATA_DIR / "olist_customers_dataset.csv"


def main() -> None:
    """Carrega os dados brutos, aplica os tratamentos e salva a base intermediaria."""
    # 1. Carrega os 3 CSVs esperados pelo load_data
    orders, items, products = load_data(ORDERS_PATH, ITEMS_PATH, PRODUCTS_PATH)

    # 2. Carrega customers separadamente para enviar ao create_dataset
    customers = import_customers(CUSTOMERS_PATH)

    # 3. Cria o dataset e as features
    dataset = create_dataset(orders, items, customers)
    dataset = create_features(dataset)

    # 4. Salva o resultado
    save_data(dataset, INTERIM_DATA_DIR)

    
X_train, X_test, y_train, y_test = split_data(data)

models = train_models(X_train, y_train)

evaluate_models(models, X_test, y_test)


def import_customers(path):
    import pandas as pd
    return pd.read_csv(path)


if __name__ == "__main__":
    main()