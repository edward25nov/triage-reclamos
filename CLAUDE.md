# Triage de reclamos

Clasificador de reclamos financieros: recibe el texto de un reclamo y devuelve el equipo que debe atenderlo.
Datos: narrativas públicas del CFPB (archivo del FOIA Reading Room). Caso de estudio del curso Product Development.

## Comandos
- `make data`: construye data/processed/ (train.parquet y val.parquet) a partir de data/raw/.
- `make train`: entrena TF-IDF + regresión logística y escribe metrics.json.
- `make test`: ejecuta las pruebas.
- Sin make: `PYTHONPATH=src python -m triage.data.build_dataset`, `PYTHONPATH=src python -m triage.models.train` y `python -m pytest`.

## Estructura
- `src/triage/data/build_dataset.py`: limpieza, deduplicación, etiquetas, muestreo y split temporal.
- `src/triage/models/train.py`: entrenamiento y métricas.
- `configs/label_map.yaml`: mapa de categorías del CFPB a clases de negocio.
- `params.yaml`: semilla, split, muestreo e hiperparámetros.
- `data/raw/`: extracto de 2023 · `data/golden/`: conjunto de prueba fijo (primer semestre de 2025).

## Reglas del modelo
- La única feature es el texto del reclamo. Nunca usar Issue, Sub-issue ni Sub-product como features: es fuga de la etiqueta.
- Sub-product solo puede usarse en label_map.yaml para construir la etiqueta.
- El split es temporal (split.val_start en params.yaml), nunca aleatorio.
- golden_test nunca se usa para entrenar.
- Ordenar por complaint_id antes de muestrear y usar la semilla de params.yaml.
