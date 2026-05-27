import os
import pickle

import runpod


def load_model():
    """Load trained model from local model directory."""
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


MODEL = load_model()

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "FamilySize", "IsAlone", "FarePerPerson",
]


def handler(event):
    """RunPod serverless handler for Titanic survival prediction.

    Expected input: {"input": {"samples": [{"Pclass": 3, "Sex": 0, "Age": 22,
        "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": 0,
        "FamilySize": 2, "IsAlone": 0, "FarePerPerson": 3.625}]}}
    Returns: {"prediction": [0]}  (0=did not survive, 1=survived)
    """
    input_data = event.get("input", {})
    samples = input_data.get("samples", [])

    if MODEL is None:
        return {"error": "Model not loaded"}

    if not samples:
        return {"error": "No samples provided"}

    for i, sample in enumerate(samples):
        missing = [f for f in FEATURES if f not in sample]
        if missing:
            return {"error": f"Sample {i} missing features: {missing}"}

    X = [[sample[f] for f in FEATURES] for sample in samples]
    predictions = MODEL.predict(X).tolist()
    return {"prediction": predictions}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
