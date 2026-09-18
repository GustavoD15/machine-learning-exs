import pandas as pd
import shap
from scipy import sparse

def prepare_data_for_shap(pipeline,X):
    #Recupera o pre-processador
    preprocessor = pipeline.named_steps['preprocessor']
    #Aplica as transformacoes
    X_transformed = preprocessor.transform(X)
    if sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()

    # Recuperar os nomes das feat
    feature_names = preprocessor.get_feature_names_out()

    #Converte para um dataframe
    X_transformed = pd.DataFrame(
        X_transformed,
          columns=feature_names,
          index=X.index,
          )
    return X_transformed

def create_explainer(pipeline):
    model = pipeline.named_steps['model']
    #Criar um explicador que entenda as previsoes
    # desse modelo
    explainer = shap.TreeExplainer(model)
    return explainer

def calculate_shap_values(explainer, X_transformed):
    #Calcula os valores de shap
    shap_values = explainer.shap_values(X_transformed)
    return shap_values