"""Entrena el baseline (TF-IDF + regresión logística) y escribe metrics.json."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from triage.data.build_dataset import asignar_clase, cargar_params, cargar_yaml, renombrar


def construir_modelo(params: dict) -> Pipeline:
    tfidf = params["features"]["tfidf"]
    modelo = params["model"]
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=tfidf["max_features"],
            ngram_range=tuple(tfidf["ngram_range"]),
            min_df=tfidf["min_df"],
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=modelo["C"], max_iter=modelo["max_iter"])),
    ])


def f1_macro(modelo: Pipeline, df: pd.DataFrame) -> float:
    return round(float(f1_score(df["clase"], modelo.predict(df["texto"]), average="macro")), 4)


def main() -> None:
    params = cargar_params()
    procesados = Path(params["data"]["processed_dir"])
    train = pd.read_parquet(procesados / "train.parquet")
    val = pd.read_parquet(procesados / "val.parquet")

    golden = renombrar(pd.read_parquet(params["data"]["golden_path"]))
    golden["texto"] = golden["texto"].fillna("")
    golden["clase"] = asignar_clase(golden, cargar_yaml(params["data"]["label_map"]))

    modelo = construir_modelo(params).fit(train["texto"], train["clase"])
    Path("models").mkdir(exist_ok=True)
    ruta_modelo = Path("models") / "model.joblib"
    joblib.dump(modelo, ruta_modelo)

    metricas = {
        "f1_macro_val": f1_macro(modelo, val),
        "f1_macro_golden": f1_macro(modelo, golden),
        "n_train": len(train),
        "n_val": len(val),
        "n_clases": int(train["clase"].nunique()),
        "tamano_modelo_mb": round(ruta_modelo.stat().st_size / 1e6, 1),
    }
    Path("metrics.json").write_text(json.dumps(metricas, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metricas, indent=2))


if __name__ == "__main__":
    main()
