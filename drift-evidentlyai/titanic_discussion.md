# Análisis de Deriva del Dato — Dataset Titanic con Evidently AI

## 1. Configuración experimental

### Dataset
El dataset crudo del Titanic contiene **891 pasajeros** con las siguientes variables:

| Variable | Tipo | Descripción |
|---|---|---|
| `Survived` | Categórica (objetivo) | 0 = No sobrevivió, 1 = Sobrevivió (342/549) |
| `Pclass` | Categórica | Clase del billete (1ª, 2ª, 3ª) |
| `Sex` | Categórica | Sexo del pasajero |
| `Embarked` | Categórica | Puerto de embarque (S=644, C=168, Q=77) |
| `Age` | Numérica | Edad (177 valores nulos, ~20%) |
| `SibSp` | Numérica | Nº de hermanos/cónyuge a bordo |
| `Parch` | Numérica | Nº de padres/hijos a bordo |
| `Fare` | Numérica | Tarifa pagada |

Se excluyeron `PassengerId`, `Name`, `Ticket` y `Cabin` por ser identificadores o tener alta cardinalidad/nulos.

### Condiciones experimentales

Se generaron **12 condiciones de división** combinando:

- **Estratificación**: Sí (por `Survived`) / No → 2 variantes
- **Proporciones train/val/test**: 60/20/20, 90/5/5, 98/1/1 → 3 variantes
- **Semilla aleatoria**: 42, 123 → 2 variantes

La referencia en los informes de deriva es siempre el **conjunto de entrenamiento**. La detección usa el umbral estándar p < 0.05 (K-S para numéricas, Z-test para categóricas).

---

## 2. Tabla resumen de resultados

| # | Estratif. | Ratio (tr/val/te) | Semilla | N train | N val | N test | Frac. deriva val | Frac. deriva test | Columna con deriva |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **Sí** | 60/20/20 | 42 | 534 | 178 | 179 | **0/8 (0%)** | **0/8 (0%)** | — |
| 1 | **Sí** | 60/20/20 | 123 | 534 | 178 | 179 | **1/8 (12.5%)** | **0/8 (0%)** | Age (val) |
| 2 | **Sí** | 90/5/5 | 42 | 801 | 45 | 45 | **0/8 (0%)** | **0/8 (0%)** | — |
| 3 | **Sí** | 90/5/5 | 123 | 801 | 45 | 45 | **0/8 (0%)** | **1/8 (12.5%)** | Embarked (test) |
| 4 | **Sí** | 98/1/1 | 42 | 873 | 9 | 9 | **0/8 (0%)** | **0/8 (0%)** | — |
| 5 | **Sí** | 98/1/1 | 123 | 873 | 9 | 9 | **1/8 (12.5%)** | **0/8 (0%)** | Age (val) |
| 6 | **No** | 60/20/20 | 42 | 534 | 178 | 179 | **0/8 (0%)** | **0/8 (0%)** | — |
| 7 | **No** | 60/20/20 | 123 | 534 | 178 | 179 | **0/8 (0%)** | **0/8 (0%)** | — |
| 8 | **No** | 90/5/5 | 42 | 801 | 45 | 45 | **1/8 (12.5%)** | **0/8 (0%)** | Embarked (val) |
| 9 | **No** | 90/5/5 | 123 | 801 | 45 | 45 | **0/8 (0%)** | **0/8 (0%)** | — |
| 10 | **No** | 98/1/1 | 42 | 873 | 9 | 9 | **0/8 (0%)** | **0/8 (0%)** | — |
| 11 | **No** | 98/1/1 | 123 | 873 | 9 | 9 | **0/8 (0%)** | **1/8 (12.5%)** | Sex (test) |

> **Nota**: "Frac. deriva" = columnas con deriva detectada / total columnas analizadas (8).  
> Reportes HTML trazables en `reports/titanic/` con nomenclatura `{estratif}_{ratio}_{semilla}_{val|test}.html`.

---

## 3. Discusión de resultados

### 3.1 Nivel general de deriva

De los 24 casos analizados (12 condiciones × 2 conjuntos), **5 muestran deriva** (≈21%), y en todos los casos afecta exactamente **1 de 8 columnas (12.5%)**. En ningún caso se detecta deriva en más de una variable simultáneamente, lo que indica que las divisiones aleatorias del dataset son en general representativas.

Las columnas que presentaron deriva son:

- **`Age`** (2 casos, condiciones 1 y 5): variable numérica con ~20% de valores nulos. Su distribución irregular —condicionada por clase, sexo y puerto— la hace sensible a variaciones muestrales. El test K-S detecta diferencias en la distribución acumulada.
- **`Embarked`** (2 casos, condiciones 3 y 8): variable categórica con distribución muy asimétrica (S=72%, C=19%, Q=9%). La clase minoritaria Q puede quedar sub o sobre-representada en muestras pequeñas.
- **`Sex`** (1 caso, condición 11): variable binaria (male=65%, female=35%). Con solo 9 muestras sin estratificación, el azar puede producir desequilibrios significativos estadísticamente.

La variable objetivo `Survived` **nunca presenta deriva**, tanto con estratificación como sin ella. Esto es esperable: la división aleatoria tiende a preservar la proporción global (38.4% supervivencia) incluso sin estratificar explícitamente, salvo en tamaños de muestra muy reducidos.

---

### 3.2 Efecto de la estratificación

La estratificación por `Survived` **no reduce la probabilidad de deriva** en las variables de entrada (features). De hecho, el número de casos con deriva es idéntico en ambas condiciones (3 casos con estratificación, 2 sin ella).

Esto es consistente con la teoría: la estratificación garantiza que la distribución de la variable objetivo se preserve entre particiones, pero **no controla la distribución del resto de variables**. Las diferencias observadas en `Age`, `Embarked` y `Sex` son consecuencia del muestreo aleatorio sobre los features, independientemente de la estratificación.

La comparación entre condiciones con/sin estratificación del mismo ratio y semilla es directa:

| Ratio | Semilla | Deriva val (c/ est.) | Deriva val (s/ est.) | Deriva test (c/ est.) | Deriva test (s/ est.) |
|---|---|---|---|---|---|
| 60/20/20 | 42 | 0% | 0% | 0% | 0% |
| 60/20/20 | 123 | 12.5% (Age) | 0% | 0% | 0% |
| 90/5/5 | 42 | 0% | 12.5% (Embarked) | 0% | 0% |
| 90/5/5 | 123 | 0% | 0% | 12.5% (Embarked) | 0% |
| 98/1/1 | 42 | 0% | 0% | 0% | 0% |
| 98/1/1 | 123 | 12.5% (Age) | 0% | 0% | 12.5% (Sex) |

No se observa un patrón sistemático: la estratificación no sistematicamente elimina ni introduce deriva en los features.

---

### 3.3 Efecto de las proporciones de división

La proporción de división afecta el **tamaño de los conjuntos** de validación y test, con consecuencias directas sobre la fiabilidad estadística de los tests de deriva:

| Ratio | N val/test | Sensibilidad del test |
|---|---|---|
| 60/20/20 | ~178 / ~179 | Alta (muestra representativa) |
| 90/5/5 | 45 / 45 | Moderada (suficiente para distribuciones simples) |
| 98/1/1 | 9 / 9 | Muy baja (estadísticamente poco fiable) |

**Con 60/20/20**, los conjuntos de val/test tienen suficiente tamaño para que los tests sean robustos. En condición 7 (no estratificado, seed=123) que muestra 0% de deriva, esto es altamente confiable.

**Con 90/5/5**, los 45 ejemplos de val/test permiten tests razonables, pero la variable `Embarked` (con su categoría Q minoritaria, solo ~4 ejemplos esperados en 45) se vuelve vulnerable a fluctuaciones. De ahí que Embarked aparezca con deriva en dos de estas condiciones.

**Con 98/1/1**, los conjuntos de 9 muestras hacen que los tests estadísticos sean prácticamente ininterpretables. Un único pasajero de más o de menos en la categoría minoritaria genera un cambio de distribución estadísticamente significativo, pero no refleja deriva real del proceso de generación de datos. Esta es la condición más problemática desde el punto de vista práctico.

En producción, usar ratios tan extremos (98/1/1) implica:
- **Consecuencias positivas**: el modelo se entrena con casi todo el dato disponible.
- **Consecuencias negativas**: los sets de validación y test son demasiado pequeños para detectar deriva de forma confiable. Un test de deriva con 9 muestras tiene bajísima potencia estadística, y la estimación de métricas de evaluación del modelo tendrá alta varianza.

---

### 3.4 Efecto de la semilla aleatoria

La semilla tiene un **impacto notable**: 4 de los 5 casos con deriva ocurren con seed=123. Esto no indica que seed=123 sea "peor", sino que el resultado particular de ese shuffle provoca que ciertos subconjuntos sean menos representativos de la distribución global.

Concretamente:
- **Seed=42**: solo muestra deriva en 1 caso (strat-no, 90/5/5, val → Embarked).
- **Seed=123**: muestra deriva en 4 casos, afectando Age, Embarked y Sex.

Este efecto confirma que con conjuntos pequeños (45 o 9 muestras), la variabilidad debida al muestreo puede ser mayor que la variabilidad "real" entre particiones. Con el ratio 60/20/20 y sin estratificación, ninguna semilla produce deriva, lo que sugiere que con suficiente tamaño muestral el efecto de la semilla se minimiza.

---

### 3.5 Asimetría entre val y test

En los 5 casos con deriva, val y test nunca presentan deriva simultáneamente en el mismo experimento. Esto es consistente con el procedimiento de división: val y test provienen de la misma pool temporal (la fracción no-train), dividida aleatoriamente al 50%. Cuando una mitad es más "extrema", la otra tiende a compensar.

---

## 4. Conclusión general

El análisis de deriva sobre el dataset Titanic bajo 12 condiciones distintas de división revela que, con divisiones razonables (60/20/20 o incluso 90/5/5), los conjuntos de entrenamiento, validación y test son estadísticamente representativos entre sí: la deriva es ausente o marginal (máximo 1 de 8 columnas) y dependiente de variaciones aleatorias de muestreo más que de diferencias estructurales reales.

Los principales factores que determinan si se detecta deriva son:

1. **Tamaño de la muestra**: el factor más crítico. Particiones muy pequeñas (9 muestras en 98/1/1) amplifican la variabilidad del muestreo y producen detecciones espúreas de deriva estadística que no reflejan cambios reales en la distribución de los datos.

2. **Variables con alta variabilidad o distribución asimétrica**: `Age` (con 20% de nulos) y `Embarked` (con categoría rara Q) son las más susceptibles a mostrar deriva aparente en muestras pequeñas.

3. **Semilla aleatoria**: determina el resultado específico del shuffle y tiene impacto, especialmente con conjuntos pequeños. No es un parámetro de diseño sino una fuente de varianza que debe tenerse en cuenta.

4. **Estratificación**: protege únicamente la distribución de la variable objetivo. No elimina la posibilidad de deriva en features, aunque tampoco la incrementa sistemáticamente.

En la práctica, para un proyecto de MLOps basado en el dataset Titanic, la división **60/20/20 con estratificación** es la configuración más robusta: proporciona conjuntos de validación y test suficientemente grandes para una detección de deriva fiable y métricas de evaluación del modelo con baja varianza. Las divisiones extremas como 98/1/1 solo serían justificables si el objetivo principal es maximizar los datos de entrenamiento y el monitoreo de deriva se realiza con otro mecanismo (e.g., datos de producción en lugar de un conjunto estático de test).

---

*Reportes HTML: `reports/titanic/` | Estadísticas: `reports/titanic/drift_stats.json` | Código: `titanic_drift_analysis.py`*
