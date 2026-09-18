import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)

from module_olist.modeling.predict import (
    load_model,
)

from module_olist.modeling.interpret import (
    prepare_data_for_shap,
    create_explainer,
    calculate_shap_values,
)

def main():
    # Load the trained model pipeline
    pipeline, model_name, threshold = load_model(
    MODELS_DIR / "best_model.joblib",
    MODELS_DIR / "metadata.json",
)

    # Load the dataset for SHAP analysis
    X = pd.read_csv(INTERIM_DATA_DIR / "X_test.csv")

    # Prepare the data for SHAP analysis
    X_transformed = prepare_data_for_shap(pipeline, X)

    # Create a SHAP explainer
    explainer = create_explainer(pipeline)

    # Calculate SHAP values
    shap_values = calculate_shap_values(explainer, X_transformed)

    # Plot SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed)
    plt.savefig(FIGURES_DIR / "shap_summary_plot.png")
    logger.info("SHAP summary plot saved.")

if __name__ == "__main__":
    main()