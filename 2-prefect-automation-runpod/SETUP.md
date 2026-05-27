# Setup — Lab B: Migración a RunPod

## Prerequisitos

- Docker Desktop instalado en Windows con integración WSL activada
- Cuenta en Docker Hub (`juankibenitez`)
- Cuenta en RunPod con créditos
- W&B artifact `titanic_processed:latest` disponible (generado en Lab A → F02a)

---

## 1. Docker Hub — Crear Personal Access Token (PAT)

1. Ve a https://hub.docker.com → tu perfil → **Account Settings**
2. Clic en **Personal Access Tokens** → **Generate new token**
3. Dale nombre (ej. `mlops-lab`) y permisos **Read & Write**
4. Copia el token y ponlo en `.env`:
   ```
   DOCKER_PAT=<tu_token_aqui>
   ```

---

## 2. RunPod — Crear Serverless Template y Endpoint

### 2a. Crear template (antes de ejecutar f01_build y f03_deploy)

1. Ve a https://www.runpod.io/console/serverless
2. Clic en **New Template**
3. Rellena:
   - **Template name:** `titanic-rf-predict`
   - **Container image:** `juankibenitez/titanic-rf-predict:latest`
   - **Container disk:** 5 GB
4. Guarda y copia el **Template ID** (aparece en la URL o en la lista)
5. Ponlo en `config/runpod.json` → `deployment.template_id`

### 2b. Crear endpoint serverless

1. En la misma sección Serverless → **New Endpoint**
2. Selecciona el template creado
3. Configura workers (min=0, max=1 para ahorrar créditos)
4. Guarda y copia el **Endpoint ID**
5. Ponlo en `config/runpod.json` → `deployment.endpoint_id`

---

## 3. Variables de entorno

Edita el fichero `.env` con todos los valores reales:

```
PREFECT_API_URL=http://127.0.0.1:4200/api
RUNPOD_API_KEY=<tu_runpod_api_key>
WANDB_API_KEY=<tu_wandb_api_key>
DOCKER_PAT=<tu_docker_hub_pat>
```

---

## 4. Ejecución de los flujos

```bash
# Terminal 1 — servidor Prefect
prefect server start

# Terminal 2 — instalar dependencias
uv sync

# Paso 1: construir y subir imagen de entrenamiento a Docker Hub
uv run python f01_build.py

# Paso 2: lanzar Pod de entrenamiento en RunPod
#   (el Pod descarga titanic_processed de W&B, entrena RF, sube titanic-rf-runpod a W&B y se para)
uv run python f02_train_pod.py

# Paso 3: desplegar modelo como endpoint serverless en RunPod
#   (descarga modelo de W&B, construye imagen predict, la sube y actualiza el endpoint)
uv run python f03_deploy.py

# O desplegar todos los flows de Prefect con un solo comando:
uv run python deploy_flows.py
```

---

## 5. Probar el endpoint serverless

Una vez desplegado, envía una petición de prueba:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "samples": [{
        "Pclass": 3, "Sex": 0, "Age": 22, "SibSp": 1, "Parch": 0,
        "Fare": 7.25, "Embarked": 0, "FamilySize": 2, "IsAlone": 0,
        "FarePerPerson": 3.625
      }]
    }
  }'
```

Respuesta esperada: `{"prediction": [0]}` (0 = no sobrevive, 1 = sobrevive)

---

## Flujos implementados

| Flow | Fichero | Descripción |
|------|---------|-------------|
| Build training image | `f01_build.py` | Construye y sube imagen Docker de entrenamiento |
| Launch training pod | `f02_train_pod.py` | Lanza Pod en RunPod que entrena el modelo y lo sube a W&B |
| Deploy serverless | `f03_deploy.py` | Construye imagen predict, la sube y actualiza endpoint RunPod |

## Nota sobre créditos RunPod

Tras ejecutar `f02_train_pod.py`, el Pod se detiene automáticamente tras 60 segundos de margen para inspección de logs. El endpoint serverless con `min_workers=0` no consume créditos cuando está inactivo.
