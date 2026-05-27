# Lab - Entrenamiento de modelo en AWS

Aplicación práctica de Amazon SageMaker sobre el dataset Titanic.  
Implementa un **Processing Job** de limpieza y feature engineering y un **Training Job** con script personalizado.

## Estructura

```
├── data/
│   └── titanic_raw.csv          # Dataset crudo (891 pasajeros)
├── src/
│   ├── preprocessing.py         # Script del Processing Job
│   └── train.py                 # Script del Training Job
└── notebook.ipynb               # Orquestador (ejecutar en SageMaker Studio)
```

## Flujo

```
titanic_raw.csv (S3)
        │
        ▼
 Processing Job          ← src/preprocessing.py
 (SKLearnProcessor)
        │
   ┌────┴────┐
train.csv  validation.csv  (S3)
        │
        ▼
  Training Job           ← src/train.py
  (SKLearn estimator)
        │
        ▼
   model.joblib (S3)
```

## Processing Job — `src/preprocessing.py`

Operaciones aplicadas sobre el dataset crudo:

| Paso | Operación |
|------|-----------|
| Drop | `PassengerId`, `Name`, `Ticket`, `Cabin` |
| Impute | `Age` → mediana, `Embarked` → moda, `Fare` → mediana |
| Encode | `Sex` → `{male:0, female:1}`, `Embarked` → `{S:0, C:1, Q:2}` |
| FE | `FamilySize = SibSp + Parch + 1`, `IsAlone`, `FarePerPerson = Fare / FamilySize` |
| Split | 80% train / 20% validation, estratificado por `Survived` |

**Salida**: CSV sin cabecera, primera columna = `Survived` (requerido por XGBoost built-in y compatible con sklearn custom).

## Training Job — `src/train.py`

- Modelo: `RandomForestClassifier` (scikit-learn)
- Hiperparámetros por defecto: `n_estimators=100`, `max_depth=8`, `random_state=42`, `class_weight=balanced`
- Métricas capturadas por SageMaker desde logs: accuracy, precision, recall, F1

**Resultados esperados (validados localmente):**

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.8045 |
| Precision | 0.7576 |
| Recall | 0.7246 |
| F1 | 0.7407 |

Features más importantes: `Sex` (30%), `Fare` (17%), `FarePerPerson` (16%), `Age` (15%), `Pclass` (8%).

## Ejecución

1. Abrir `notebook.ipynb` en **SageMaker Studio** (dominio configurado con `LabRole`)
2. Ejecutar celda a celda — cada job lanza su propia instancia efímera en AWS
3. El Processing Job usa `ml.m5.xlarge` y el Training Job usa `ml.m5.large` con **Spot Instances**

## Prerequisitos

- Dominio SageMaker activo con rol `LabRole`
- Bucket S3 por defecto disponible (se crea automáticamente con `sess.default_bucket()`)
