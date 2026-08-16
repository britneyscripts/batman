import os
import shutil
import kagglehub


def copy_files(source_path: str, target_dir: str, prefix: str = ""):
    """Copia arquivos do cache do kagglehub para data/raw com prefixo opcional."""
    for file_name in os.listdir(source_path):
        src = os.path.join(source_path, file_name)
        dst = os.path.join(target_dir, f"{prefix}{file_name}")
        if os.path.isfile(src):
            shutil.copy(src, dst)
            print(f"  [OK] Copiado: {file_name} -> {dst}")


def download_all_datasets():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("Iniciando download de todos os datasets do Kaggle via kagglehub...\n")

    # 1. Prophet / Time Series: Store Sales (Competição)
    print("1. Baixando 'store-sales-time-series-forecasting'...")
    try:
        p1 = kagglehub.competition_download(
            "store-sales-time-series-forecasting"
        )
        copy_files(p1, raw_dir, prefix="store_sales_")
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar 'store-sales-time-series-forecasting': {e}")

    # 2. Prophet / Time Series: Hourly Energy Consumption
    print("\n2. Baixando 'hourly-energy-consumption'...")
    try:
        p3 = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
        copy_files(p3, raw_dir, prefix="energy_")
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar 'hourly-energy-consumption': {e}")

    # 3. MMM Multichannel: Nafees MMM Dataset
    print("\n3. Baixando 'mmm-dataset'...")
    try:
        p4 = kagglehub.dataset_download("nafees2006/mmm-dataset")
        copy_files(p4, raw_dir, prefix="mmm_nafees_")
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar 'mmm-dataset': {e}")

    # 4. MMM Multichannel: Matt Walentosky Demo Dataset
    print("\n4. Baixando 'mmmdemodataset'...")
    try:
        p5 = kagglehub.dataset_download("mattwalentosky/mmmdemodataset")
        copy_files(p5, raw_dir, prefix="mmm_walentosky_")
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar 'mmmdemodataset': {e}")

    # 5. MMM Traditional + Digital: Retail / CPG Market Mix Dataset
    print("\n5. Baixando 'marrket-mix-dataset'...")
    try:
        p6 = kagglehub.dataset_download("veer06b/marrket-mix-dataset")
        copy_files(p6, raw_dir, prefix="mmm_retail_")
    except Exception as e:
        print(f"  [ERRO] Falha ao baixar 'marrket-mix-dataset': {e}")

    print("\n Download e sincronização de todas as bases concluídos em 'data/raw/'!")


if __name__ == "__main__":
    download_all_datasets()