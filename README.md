# Triage de reclamos

Caso de estudio del curso **Product Development** (Maestría en Data Science, Universidad Galileo).

Un servicio que recibe el texto de un reclamo financiero y lo asigna al equipo responsable.
Datos: narrativas públicas de reclamos del CFPB.

```bash
pip install -r requirements.txt
make data    # construye data/processed/
make train   # entrena el baseline y escribe metrics.json
make test    # ejecuta las pruebas
```
