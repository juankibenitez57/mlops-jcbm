# Documento de Diseño — Lab A: Workflow con Prefect y Weights & Biases

**Dataset:** Titanic (Kaggle)  
**Herramientas:** Prefect 3.x · Weights & Biases (W&B) · scikit-learn · Python 3.13  
**Proyecto W&B:** `titanic-mlops`

---

## 1. Visión general

Se implementan dos patrones de flujo sobre el mismo problema de clasificación binaria (predicción de supervivencia en el Titanic):

| Patrón | Fichero | Flujos |
|--------|---------|--------|
| Monolito acoplado | `f01_monolith.py` | 1 flow · 6 tasks |
| Desacoplado por artefacto | `f02_decoupled.py` | 2 flows · 8 tasks |

Ambos flujos están orquestados con Prefect y persisten métricas, datasets y modelos en W&B como artefactos versionados.

---

## 2. Patrón 1 — Flujo monolítico (F01)

### Diagrama de etapas

```
[W&B: titanic_raw]
        │
        ▼
┌─────────────────────────────┐
│  FLOW: F01 — Monolith       │
│                             │
│  1. load_dataset            │  ← descarga artefacto raw de W&B
│  2. process_dataset         │  ← limpieza + feature engineering
│  3. split_dataset           │  ← train/test split estratificado
│  4. train_model             │  ← entrena RandomForestClassifier
│  5. evaluate_model          │  ← métricas → W&B run.log()
│  6. register_model          │  ← serializa modelo → W&B artifact
└─────────────────────────────┘
        │
        ▼
[W&B: titanic-rf-monolith (artifact)]
[W&B: run metrics — accuracy, precision, recall, F1]
```

### Descripción de cada etapa

| # | Task | Entrada | Salida | Responsabilidad |
|---|------|---------|--------|-----------------|
| 1 | `load_dataset` | W&B artifact ref | `DataFrame` | Descarga el CSV crudo del registro de artefactos W&B |
| 2 | `process_dataset` | `DataFrame` crudo | `DataFrame` limpio | Imputa nulos, codifica categóricas, genera FamilySize / IsAlone / FarePerPerson |
| 3 | `split_dataset` | `DataFrame` limpio | X_train, X_test, y_train, y_test | División 80/20 con estratificación por clase |
| 4 | `train_model` | X_train, y_train | `RandomForestClassifier` | Ajusta el modelo con hiperparámetros del config |
| 5 | `evaluate_model` | modelo, X_test, y_test | `dict` de métricas | Calcula y registra accuracy, precision, recall, F1 en W&B |
| 6 | `register_model` | modelo | — | Serializa el modelo como pickle y lo publica como artefacto W&B |

### Criterios de diseño del flujo monolítico

- **Un único contexto W&B:** todas las etapas comparten el mismo `wandb.run`, lo que facilita la trazabilidad completa de un experimento en una sola vista.
- **Simplicidad de ejecución:** adecuado para experimentos rápidos donde no se requiere reutilizar etapas parciales.
- **Datos en memoria:** los DataFrames y el modelo se pasan directamente entre tasks, sin serialización intermedia.

---

## 3. Patrón 2 — Flujos desacoplados (F02)

### Diagrama de etapas

```
[W&B: titanic_raw]
        │
        ▼
┌──────────────────────────────┐
│  FLOW F02a — Process data    │
│                              │
│  1. load_dataset             │  ← descarga artefacto raw
│  2. process_dataset          │  ← limpieza + feature engineering
│  3. persist_processed_dataset│  ← guarda CSV procesado → W&B artifact
└──────────────────────────────┘
        │
        ▼
[W&B: titanic_processed (artifact)]  ◄── punto de desacoplamiento
        │
        ▼
┌──────────────────────────────┐
│  FLOW F02b — Train model     │
│                              │
│  4. load_processed_dataset   │  ← carga artefacto procesado de W&B
│  5. split_dataset            │  ← train/test split estratificado
│  6. train_model              │  ← entrena RandomForestClassifier
│  7. evaluate_model           │  ← métricas → W&B run.log()
│  8. register_model           │  ← serializa modelo → W&B artifact
└──────────────────────────────┘
        │
        ▼
[W&B: titanic-rf-decoupled (artifact)]
[W&B: run metrics — accuracy, precision, recall, F1]
```

### Descripción de cada etapa

| # | Task | Flow | Entrada | Salida | Responsabilidad |
|---|------|------|---------|--------|-----------------|
| 1 | `load_dataset` | F02a | W&B artifact ref | `DataFrame` | Descarga el CSV crudo |
| 2 | `process_dataset` | F02a | `DataFrame` crudo | `DataFrame` limpio | Limpieza y feature engineering |
| 3 | `persist_processed_dataset` | F02a | `DataFrame` limpio | — | Persiste el dataset procesado como artefacto W&B (`titanic_processed`) |
| 4 | `load_processed_dataset` | F02b | W&B artifact ref | `DataFrame` | Carga el artefacto procesado (desacopla F02a de F02b) |
| 5 | `split_dataset` | F02b | `DataFrame` | X_train, X_test, y_train, y_test | División 80/20 estratificada |
| 6 | `train_model` | F02b | X_train, y_train | modelo | Ajusta RandomForestClassifier |
| 7 | `evaluate_model` | F02b | modelo, X_test, y_test | métricas | Registra métricas en W&B |
| 8 | `register_model` | F02b | modelo | — | Publica el modelo como artefacto W&B |

### Criterios de diseño de los flujos desacoplados

- **Separación de responsabilidades (SRP):** procesamiento de datos y entrenamiento del modelo son dominios distintos con ciclos de vida diferentes. El dataset cambia raramente; el modelo puede reentrenarse varias veces con distintos hiperparámetros.
- **W&B artifact como contrato:** `titanic_processed` es la interfaz entre F02a y F02b. Cualquier consumidor puede descargar ese artefacto independientemente.
- **Reejecutabilidad parcial:** si solo cambia el modelo (hiperparámetros, arquitectura), se puede lanzar únicamente F02b sin repetir la limpieza de datos.
- **Trazabilidad por separado:** cada flow genera su propio W&B run con `job_type` diferente (`data-processing` vs `model-train`), facilitando el filtrado en el panel W&B.
- **Escalabilidad:** en producción, F02a puede ejecutarse en un schedule diario (cuando llegan nuevos datos) y F02b en un schedule separado o bajo demanda.

---

## 4. Criterios de división en tasks (@task)

Cada función decorada con `@task` satisface los siguientes principios:

| Criterio | Descripción |
|----------|-------------|
| **Responsabilidad única** | Cada task hace una única cosa: descargar, limpiar, dividir, entrenar, evaluar o registrar |
| **Reintento independiente** | Prefect puede reintentar una task fallida sin repetir las anteriores |
| **Observabilidad** | Cada task aparece con nombre propio en la UI de Prefect, con su estado, duración y logs |
| **Frontera de I/O** | Las tasks solo se comunican a través de sus parámetros de entrada/salida, sin estado compartido |
| **Granularidad ML estándar** | Las divisiones siguen las etapas naturales de un pipeline ML: ingesta → preprocesado → split → entrenamiento → evaluación → registro |

---

## 5. Configuración y despliegue

### Variables de entorno (`.env`)

```
PREFECT_API_URL=http://127.0.0.1:4200/api
WANDB_API_KEY=<tu_clave_wandb>
```

### Opción A — Servidor local (sin Docker)

```bash
# Terminal 1: arrancar servidor Prefect
prefect server start

# Terminal 2: desplegar ambos flujos
python deploy_flows.py
```

### Opción B — Docker Compose

```bash
docker compose up -d
python deploy_flows.py
```

### Ejecución directa (sin despliegue)

```bash
# Solo el flujo monolítico
python f01_monolith.py

# Ambos flujos desacoplados en secuencia
python f02_decoupled.py
```

---

## 6. Resultados obtenidos

| Métrica | F01 Monolito | F02b Desacoplado |
|---------|-------------|-----------------|
| Accuracy | 80.4% | 80.4% |
| Precision | 75.8% | 75.8% |
| Recall | 72.5% | 72.5% |
| F1-score | 74.1% | 74.1% |

**Modelo:** RandomForestClassifier · n_estimators=100 · max_depth=8 · class_weight=balanced  
**Split:** 80% train / 20% test · stratify=Survived · random_state=42

**Feature engineering aplicado:**
- Imputación: Age (mediana), Embarked (moda), Fare (mediana)
- Codificación: Sex (male=0, female=1), Embarked (S=0, C=1, Q=2)
- Nuevas features: FamilySize = SibSp + Parch + 1, IsAlone, FarePerPerson
