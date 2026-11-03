from flask import Flask, jsonify, request
from app.features import prediction_frame
from app.model import load_artifacts
from app.logging_config import get_logger

def create_app(model=None, metadata=None, logger=None):
    app = Flask(__name__)
    logger = logger or get_logger()
    if model is None or metadata is None:
        try:
            model, metadata = load_artifacts()
        except FileNotFoundError:
            model, metadata = None, {"countries": []}

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": model is not None})

    @app.post("/predict")
    def predict():
        if model is None:
            return jsonify({"error": "Model is not trained. Run python -m scripts.train first."}), 503
        body = request.get_json(silent=True) or {}
        target_date = body.get("target_date")
        country = body.get("country")
        if not target_date:
            return jsonify({"error": "target_date is required (YYYY-MM-DD)."}), 400
        try:
            if country:
                X = prediction_frame(target_date, country)
                value = float(model.predict(X)[0])
                logger.info("prediction country=%s date=%s", country, target_date)
                return jsonify({"target_date": target_date, "country": country, "predicted_revenue": value})
            countries = metadata.get("countries", [])
            if not countries:
                return jsonify({"error": "No countries available in model metadata."}), 400
            rows = []
            for item in countries:
                X = prediction_frame(target_date, item)
                rows.append({"country": item, "predicted_revenue": float(model.predict(X)[0])})
            total = sum(row["predicted_revenue"] for row in rows)
            logger.info("aggregate prediction countries=%d date=%s", len(rows), target_date)
            return jsonify({"target_date": target_date, "countries": rows, "total_predicted_revenue": total})
        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
