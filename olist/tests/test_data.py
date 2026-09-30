import numpy as np
import pandas as pd

from module_olist.modeling.active_learning import run_active_learning


def test_active_learning_reveals_five_labels_per_iteration():
    sample_count = 60
    y = pd.Series(np.tile([0, 1], sample_count // 2))
    X = pd.DataFrame(
        {
            "promised_days": np.linspace(1, 30, sample_count),
            "item_count": np.ones(sample_count),
            "seller_count": np.ones(sample_count),
            "total_price": np.linspace(10, 500, sample_count),
            "total_freight": np.linspace(1, 50, sample_count),
            "purchase_month": np.tile(np.arange(1, 13), 5),
            "purchase_weekday": np.tile(np.arange(7), 9)[:sample_count],
            "purchase_hour": np.tile(np.arange(24), 3)[:sample_count],
            "customer_state": np.tile(["SP", "RJ", "MG"], 20),
        }
    )

    result = run_active_learning(
        X,
        y,
        initial_labeled=10,
        query_size=5,
        max_iterations=4,
        pool_size=sample_count,
    )

    assert result.history["labeled_count"].tolist() == [10, 15, 20, 25, 30]
    assert len(result.selections) == 20
    assert len(result.labeled_indices) == 30
