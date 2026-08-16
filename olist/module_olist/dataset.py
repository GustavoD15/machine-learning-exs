from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from loguru import logger
import typer

from module_olist.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()

# -----------------------------------------------------------------------------
# 1. Definição dos 3 Testes no Pandera
# -----------------------------------------------------------------------------
orders_schema = pa.DataFrameSchema(
    columns={
        # TESTE 1 [Unicidade & Completude]: Chave primária única e não nula
        "order_id": pa.Column(
            str, 
            nullable=False, 
            unique=True, 
            name="1. Unicidade de order_id"
        ),
        
        # TESTE 2 [Validade de Domínio]: Apenas status válidos de negócio
        "order_status": pa.Column(
            str,
            nullable=False,
            checks=pa.Check.isin([
                "delivered", "shipped", "canceled", "invoiced", 
                "processing", "unavailable", "approved", "created"
            ]),
            name="2. Validade do Status do Pedido"
        ),
    },
    # TESTE 3 [Consistência Temporal]: Entrega >= Compra
    checks=[
        pa.Check(
            lambda df: (
                df["order_delivered_customer_date"].isna() | 
                (pd.to_datetime(df["order_delivered_customer_date"]) >= pd.to_datetime(df["order_purchase_timestamp"]))
            ),
            name="3. Consistência: Entrega vs Compra",
            error="Entrega registrada antes da data de compra!"
        )
    ],
    coerce=True
)

# -----------------------------------------------------------------------------
# 2. Comando CLI Typer
# -----------------------------------------------------------------------------
@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "olist_orders_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "olist_orders_validated.csv",
    report_path: Path = REPORTS_DIR / "quality_errors_orders.csv",
):
    logger.info(f"Lendo dataset bruto de: {input_path}")
    
    if not input_path.exists():
        logger.error(f"Arquivo não encontrado: {input_path}")
        raise typer.Exit(code=1)

    df_orders = pd.read_csv(input_path)
    logger.info(f"Iniciando validação com 3 testes do Pandera em {len(df_orders)} registros...")

    try:
        # Executa a validação
        validated_df = orders_schema.validate(df_orders, lazy=True)
        logger.success("✅ 100% dos dados passaram nos 3 testes de qualidade!")
        
        # Salva o dataset limpo/validado na pasta processed
        validated_df.to_csv(output_path, index=False)
        logger.success(f"Dataset processado salvo em: {output_path}")

    except pa.errors.SchemaErrors as err:
        logger.warning(f"X Falhas de qualidade detectadas pelo Pandera! ({len(err.failure_cases)} violações)")
        
        # Salva o relatório de falhas em reports/
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        err.failure_cases.to_csv(report_path, index=False)
        logger.warning(f"Relatório detalhado das falhas salvo em: {report_path}")


if __name__ == "__main__":
    app()