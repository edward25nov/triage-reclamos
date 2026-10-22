"""Construye el dataset procesado: data/processed/train.parquet y val.parquet."""

from pathlib import Path

import pandas as pd
import yaml

# Columnas del CFPB que se usan y su nombre dentro del proyecto.
COLUMNAS = {
    "Complaint ID": "complaint_id",
    "Date received": "fecha",
    "Product": "producto",
    "Sub-product": "subproducto",
    "Consumer complaint narrative": "texto",
}


def cargar_yaml(ruta: str) -> dict:
    return yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))


def cargar_params(ruta: str = "params.yaml") -> dict:
    return cargar_yaml(ruta)


def renombrar(df: pd.DataFrame) -> pd.DataFrame:
    """Pasa de los nombres del CFPB a los del proyecto."""
    df = df.rename(columns=COLUMNAS)
    for col in COLUMNAS.values():
        if col not in df.columns:
            df[col] = None
    return df[list(COLUMNAS.values())]


def asignar_clase(df: pd.DataFrame, label_map: dict) -> pd.Series:
    """Clase de negocio a partir del producto (y, si hay reglas, del subproducto)."""
    por_producto = {p: c for c, productos in label_map["clases"].items() for p in productos}
    clase = df["producto"].map(por_producto).fillna(label_map.get("clase_por_defecto", "otros"))
    por_subproducto = label_map.get("por_subproducto") or {}
    if por_subproducto:
        regla = df["subproducto"].map(por_subproducto)
        clase = regla.where(regla.notna(), clase)
    return clase


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Quita narrativas vacías y duplicados, y ordena por complaint_id."""
    df = df[df["texto"].fillna("").str.strip().str.len() > 0].copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", format="mixed")
    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce")
    df = df.dropna(subset=["fecha", "complaint_id"])
    df["complaint_id"] = df["complaint_id"].astype("int64")
    df = df.sort_values("complaint_id").drop_duplicates(subset="complaint_id")
    texto_normalizado = df["texto"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    return df[~texto_normalizado.duplicated()].reset_index(drop=True)


def muestrear(df: pd.DataFrame, max_por_clase: int, seed: int) -> pd.DataFrame:
    """Muestra con un tope por clase. Determinista: misma semilla, misma muestra."""
    mezcla = df.sort_values("complaint_id").sample(frac=1, random_state=seed)
    return mezcla.groupby("clase").head(max_por_clase).sort_values("complaint_id").reset_index(drop=True)


def leer_crudos(raw_dir: str) -> pd.DataFrame:
    archivos = sorted(Path(raw_dir).glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos .csv en {raw_dir}")
    partes = [pd.read_csv(a, dtype=str, usecols=lambda c: c in COLUMNAS) for a in archivos]
    return renombrar(pd.concat(partes, ignore_index=True))


def construir(params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_map = cargar_yaml(params["data"]["label_map"])
    df = limpiar(leer_crudos(params["data"]["raw_dir"]))
    df["clase"] = asignar_clase(df, label_map)

    corte = pd.Timestamp(params["split"]["val_start"])
    seed = params["seed"]
    train = muestrear(df[df["fecha"] < corte], params["sample"]["max_por_clase_train"], seed)
    val = muestrear(df[df["fecha"] >= corte], params["sample"]["max_por_clase_val"], seed)
    columnas = ["complaint_id", "fecha", "texto", "clase"]
    return train[columnas], val[columnas]


def main() -> None:
    params = cargar_params()
    train, val = construir(params)
    destino = Path(params["data"]["processed_dir"])
    destino.mkdir(parents=True, exist_ok=True)
    train.to_parquet(destino / "train.parquet", index=False)
    val.to_parquet(destino / "val.parquet", index=False)
    print(f"train: {len(train):,} filas · val: {len(val):,} filas")
    print(train["clase"].value_counts().to_string())


if __name__ == "__main__":
    main()
