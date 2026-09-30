from dataclasses import dataclass

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.semi_supervised import LabelSpreading

from module_olist.config import INTERIM_DATA_DIR
from module_olist.modeling.pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from module_olist.modeling.split import split_data


@dataclass
class ActiveLearningResult:
    history: pd.DataFrame
    selections: pd.DataFrame
    labeled_indices: np.ndarray


def _create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def run_active_learning(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial_labeled: int = 10,
    query_size: int = 5,
    max_iterations: int = 4,
    pool_size: int = 1_000,
    random_state: int = 42,
) -> ActiveLearningResult:
    """Simulate uncertainty sampling with Label Spreading on a bounded data pool."""
    if len(X) != len(y):
        raise ValueError("X e y devem conter o mesmo número de observações.")
    if initial_labeled < 2 or query_size < 1 or max_iterations < 0 or pool_size < 2:
        raise ValueError("Parâmetros devem permitir ao menos duas amostras e rótulos válidos.")
    if y.nunique() < 2:
        raise ValueError("O aprendizado ativo exige pelo menos duas classes em y.")

    pool_count = min(pool_size, len(X))
    if pool_count < initial_labeled:
        raise ValueError("A amostra disponível é menor que initial_labeled.")

    all_positions = np.arange(len(X))
    if pool_count < len(X):
        pool_positions, _ = train_test_split(
            all_positions,
            train_size=pool_count,
            random_state=random_state,
            stratify=y,
        )
    else:
        pool_positions = all_positions

    X_pool = X.iloc[pool_positions].reset_index(drop=True)
    y_pool = y.iloc[pool_positions].to_numpy()
    source_indices = np.asarray(X.index)[pool_positions]

    labeled_positions, _ = train_test_split(
        np.arange(pool_count),
        train_size=initial_labeled,
        random_state=random_state,
        stratify=y_pool,
    )
    unlabeled_mask = np.ones(pool_count, dtype=bool)
    unlabeled_mask[labeled_positions] = False

    features = _create_preprocessor().fit_transform(X_pool)
    history = []
    selections = []

    for iteration in range(max_iterations + 1):
        labels = np.full(pool_count, -1, dtype=y_pool.dtype)
        labels[labeled_positions] = y_pool[labeled_positions]

        model = LabelSpreading(gamma=0.25, max_iter=20)
        model.fit(features, labels)

        unlabeled_positions = np.flatnonzero(unlabeled_mask)
        predictions = model.transduction_[unlabeled_positions]
        score = (
            accuracy_score(y_pool[unlabeled_positions], predictions)
            if len(unlabeled_positions)
            else np.nan
        )
        history.append(
            {
                "iteration": iteration,
                "labeled_count": len(labeled_positions),
                "unlabeled_count": len(unlabeled_positions),
                "unlabeled_accuracy": score,
            }
        )
        logger.info(
            "Iteração {}: {} rotulados, {} não rotulados, acurácia simulada {:.3f}",
            iteration,
            len(labeled_positions),
            len(unlabeled_positions),
            score,
        )
        if len(unlabeled_positions):
            logger.info(
                "Desempenho nas amostras ainda não rotuladas:\n{}",
                classification_report(
                    y_pool[unlabeled_positions],
                    predictions,
                    zero_division=0,
                ),
            )

        if iteration == max_iterations or not len(unlabeled_positions):
            break

        uncertainties = entropy(
            model.label_distributions_[unlabeled_positions],
            axis=1,
        )
        selection_order = np.argsort(uncertainties)[::-1]
        selected_positions = unlabeled_positions[
            selection_order[: min(query_size, len(unlabeled_positions))]
        ]

        for position in selected_positions:
            selections.append(
                {
                    "iteration": iteration + 1,
                    "sample_index": source_indices[position],
                    "uncertainty": uncertainties[
                        np.flatnonzero(unlabeled_positions == position)[0]
                    ],
                    "predicted_label": model.transduction_[position],
                    "oracle_label": y_pool[position],
                }
            )
        unlabeled_mask[selected_positions] = False
        labeled_positions = np.concatenate((labeled_positions, selected_positions))

    return ActiveLearningResult(
        history=pd.DataFrame(history),
        selections=pd.DataFrame(selections),
        labeled_indices=source_indices[labeled_positions],
    )


def plot_active_learning_results(result: ActiveLearningResult) -> None:
    if result.selections.empty:
        logger.warning("Nenhuma amostra foi selecionada para visualização.")
        return

    iterations = result.selections["iteration"].unique()
    figure, axes = plt.subplots(
        len(iterations),
        1,
        figsize=(11, max(3, 2.8 * len(iterations))),
        squeeze=False,
    )
    for axis, iteration in zip(axes.flat, iterations):
        selected = result.selections.loc[result.selections["iteration"].eq(iteration)].sort_values(
            "uncertainty"
        )
        axis.barh(
            selected["sample_index"].astype(str),
            selected["uncertainty"],
            color="#287d74",
        )
        for row, (_, sample) in enumerate(selected.iterrows()):
            axis.text(
                sample["uncertainty"],
                row,
                f"  prev.: {sample['predicted_label']} | rótulo: {sample['oracle_label']}",
                va="center",
                fontsize=8,
            )
        axis.set_title(f"Consulta {iteration}: exemplos mais incertos")
        axis.set_xlabel("Entropia da distribuição de rótulos")
        axis.set_ylabel("Índice da amostra")

    figure.tight_layout()
    plt.show()


def main() -> None:
    dataset_path = INTERIM_DATA_DIR / "orders_dataset_refined.csv" / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset refinado não encontrado em {dataset_path}. "
            "Execute primeiro o pipeline de preparação dos dados."
        )

    data = pd.read_csv(dataset_path)
    X_train, _, y_train, _ = split_data(data)
    result = run_active_learning(X_train, y_train)
    logger.info("Resumo das iterações:\n{}", result.history.to_string(index=False))
    plot_active_learning_results(result)


if __name__ == "__main__":
    main()
