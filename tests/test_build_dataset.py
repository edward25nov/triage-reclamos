import pandas as pd

from triage.data.build_dataset import asignar_clase, limpiar, muestrear

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
