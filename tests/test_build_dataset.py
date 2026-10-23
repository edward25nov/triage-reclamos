import pandas as pd
import yaml

from triage.data.build_dataset import asignar_clase, construir, leer_landing, limpiar, muestrear

LABEL_MAP = {
    "clases": {"tarjetas": ["Credit card"], "cobranza": ["Debt collection"]},
    "clase_por_defecto": "otros",
    "por_subproducto": {},
}


def datos(**columnas) -> pd.DataFrame:
    base = {"complaint_id": ["3", "1", "2"], "fecha": ["2023-01-03", "2023-01-01", "2023-01-02"],
            "producto": ["Credit card", "Debt collection", "Mortgage"],
            "subproducto": [None, None, None], "texto": ["a", "b", "c"]}
    base.update(columnas)
    return pd.DataFrame(base)


def test_limpiar_quita_vacios_y_ordena():
    df = limpiar(datos(texto=["hola", "  ", "adios"]))
    assert list(df["complaint_id"]) == [2, 3]


def test_limpiar_quita_textos_duplicados():
    df = limpiar(datos(texto=["Mismo  texto", "mismo texto", "otro"]))
    assert len(df) == 2
    assert 1 in set(df["complaint_id"])  # se conserva el más antiguo


def test_asignar_clase_usa_mapa_y_clase_por_defecto():
    clases = asignar_clase(datos(), LABEL_MAP)
    assert list(clases) == ["tarjetas", "cobranza", "otros"]


def test_subproducto_tiene_prioridad():
    label_map = {**LABEL_MAP, "por_subproducto": {"Gift card": "prepago"}}
    df = datos(subproducto=["Gift card", None, None])
    assert asignar_clase(df, label_map).iloc[0] == "prepago"


def test_muestrear_respeta_tope_y_es_determinista():
    df = pd.DataFrame({"complaint_id": range(100), "clase": ["a"] * 80 + ["b"] * 20})
    m1 = muestrear(df, max_por_clase=10, seed=42)
    m2 = muestrear(df, max_por_clase=10, seed=42)
    assert m1["clase"].value_counts().max() == 10
    assert m1.equals(m2)


def reclamo(id_, fecha, texto, producto="Credit card"):
    return {"Complaint ID": id_, "Date received": fecha, "Product": producto,
            "Sub-product": None, "Consumer complaint narrative": texto}


def escribir_dia(landing, nombre, filas):
    landing.mkdir(exist_ok=True)
    pd.DataFrame(filas).to_parquet(landing / f"{nombre}.parquet", index=False)


def test_leer_landing_une_los_parquet_de_cada_dia(tmp_path):
    landing = tmp_path / "landing"
    escribir_dia(landing, "2025-01-01", [reclamo("10", "2025-01-01", "uno")])
    escribir_dia(landing, "2025-01-02", [reclamo("11", "2025-01-02", "dos"), reclamo("12", "2025-01-02", "tres")])
    df = leer_landing(str(landing))
    assert list(df["complaint_id"]) == ["10", "11", "12"]
    assert list(df.columns) == ["complaint_id", "fecha", "producto", "subproducto", "texto"]


def test_leer_landing_sin_archivos_devuelve_vacio(tmp_path):
    assert leer_landing(str(tmp_path / "no_existe")).empty


def test_construir_suma_raw_y_landing(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([reclamo("1", "2023-01-01", "texto raw")]).to_csv(raw / "a.csv", index=False)
    escribir_dia(tmp_path / "landing", "2023-12-01", [reclamo("2", "2023-12-01", "texto landing")])
    label_map = tmp_path / "label_map.yaml"
    label_map.write_text(yaml.safe_dump({"clases": {"tarjetas": ["Credit card"]}}), encoding="utf-8")
    params = {"seed": 42, "split": {"val_start": "2023-11-01"}, "sample": {"max_por_clase_train": 10, "max_por_clase_val": 10},
              "data": {"raw_dir": str(raw), "landing_dir": str(tmp_path / "landing"), "label_map": str(label_map)}}
    train, val = construir(params)
    assert list(train["complaint_id"]) == [1]
    assert list(val["complaint_id"]) == [2]
