# App de Streamlit: oferta, demanda y bienestar

## Archivos
- `app.py`: aplicación principal
- `requirements.txt`: dependencias mínimas

## Cómo correrla
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Qué hace
- Calibra oferta y demanda en un equilibrio base `(P*, Q*)`
- Permite modificar `|ε_d|` y `ε_s`
- Evalúa bienestar bajo:
  - precio máximo
  - precio mínimo
  - impuesto específico
- Muestra:
  - cantidades y precios de equilibrio
  - CS, PS, recaudación, TS y DWL
  - tablas para distintos niveles de intervención
  - gráficos comparativos en dos columnas

## Nota sobre elasticidades extremas
La app permite usar elasticidades exactas iguales a `0` o `∞`.
En esos casos, algunos excedentes pueden dejar de ser finitos bajo la parametrización lineal límite.
La app lo reporta como `∞` o `N/D` en lugar de inventar números artificiales.
