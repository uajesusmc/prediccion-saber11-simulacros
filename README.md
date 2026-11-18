# Predicción de Saber 11° a partir de simulacros externos

Código, datos anonimizados y artículo del proyecto **«¿Pueden los simulacros de una empresa externa anticipar el resultado de la Saber 11°? Modelo predictivo y regla de priorización para una institución educativa del Atlántico»**.

**Autores:** Jesús Molina Cantillo (estudiante), John González Veloza y Julián Bernal Salamanca (profesores).
Especialización en Estadística Aplicada, Facultad de Ingeniería y Ciencias Básicas, Fundación Universitaria Los Libertadores (Bogotá, Colombia).

## Resumen

La Institución Educativa Técnica Agrícola (I.E.T.A.) Juan Domínguez Romero de Caracolí (Malambo, Atlántico) aplica cada año dos simulacros de una empresa externa (S1 y S2) antes de Saber 11° y con ellos decide a quién reforzar. Este proyecto evalúa si esos simulacros anticipan el puntaje global oficial y propone una regla para priorizar el refuerzo.

- Entrenamiento con la cohorte 2025 (n = 45) y evaluación con la cohorte 2026 (n = 52).
- Modelo elegido: regresión lineal y logística con el promedio de los dos simulacros, S̄ = (S1 + S2)/2.
- Ecuación: **SB11 estimado = 45,7 + 0,906 · S̄** (MAE en Test de 18,1 puntos, frente a 27,7 de la línea base).
- Clasificación del riesgo (cuartil inferior): AUC de 0,87 en Test y sensibilidad de 1,00. Supera a la regla de la empresa (nivel I/1) en la comparación pareada (McNemar exacta, p = 0,004).
- Regla de prioridades: **alta** si S̄ ≤ 182,5; **media** si 182,5 < S̄ ≤ 202,4; **baja** si S̄ > 202,4.

## Estructura

| Carpeta | Contenido |
| --- | --- |
| `datos/` | Bases de modelado (MDT) anonimizadas de 2025 y 2026. Cada hoja `LEEME` y `DICCIONARIO` describe las variables |
| `analisis/` | `Anexo_tecnico_modelo_Saber11.ipynb` (modelos, umbrales, figuras de los anexos), `incertidumbre_bootstrap_mcnemar.py` (intervalos de Clopper–Pearson, bootstrap y prueba de McNemar) e `interpretabilidad_coeficientes.py` (coeficientes, intervalos y valores p) |
| `herramienta/` | Hoja de cálculo para aplicar la regla de prioridades a una cohorte nueva |
| `articulo/` | Artículo en LaTeX (`main.tex`), figuras (`img/`) y PDF compilado |

## Cómo reproducir

```bash
pip install -r requirements.txt

cd analisis
jupyter notebook Anexo_tecnico_modelo_Saber11.ipynb     # lee las bases desde ../datos/
python incertidumbre_bootstrap_mcnemar.py ../datos
python interpretabilidad_coeficientes.py ../datos
```

El cuaderno fija la semilla aleatoria en 2025 y guarda las figuras en `analisis/fig/`.

Para compilar el artículo, abre `articulo/main.tex` en TeXstudio u Overleaf y compila dos veces con pdfLaTeX.

## Datos y ética

- Las bases no contienen nombres ni documentos de identidad: cada estudiante tiene un código (E001, E002, …). La tabla que vincula códigos y nombres está bajo custodia de la institución y **no** forma parte de este repositorio.
- El rector de la institución autorizó el uso de los datos en esta investigación, con la condición de que no apareciera el nombre de ningún estudiante.
- El tratamiento de los datos sigue la Ley Estatutaria 1581 de 2012 de Colombia sobre protección de datos personales.
- Los datos se comparten solo con fines académicos y de reproducibilidad. No intentes reidentificar a los estudiantes.

## Licencia

El código se distribuye bajo la licencia MIT (ver `LICENSE`). Los datos y el artículo se comparten para uso académico, citando la fuente.

## Cómo citar

Molina Cantillo, J., González Veloza, J., y Bernal Salamanca, J. (2026). *¿Pueden los simulacros de una empresa externa anticipar el resultado de la Saber 11°? Modelo predictivo y regla de priorización para una institución educativa del Atlántico*. Especialización en Estadística Aplicada, Fundación Universitaria Los Libertadores.
