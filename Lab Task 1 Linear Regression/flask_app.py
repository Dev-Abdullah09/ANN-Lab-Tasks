"""
flask_app.py
------------
Flask user interface for the two Linear Regression models.

Run:  python flask_app.py      ->  http://127.0.0.1:5000
"""

from flask import Flask, abort, render_template, request, send_from_directory

from regression_core import DATASETS, get_bundle, predict

app = Flask(__name__)


def feature_info(bundle):
    """Information needed to draw one input box per feature."""
    df = bundle["df"]
    info = []
    for f in bundle["features"]:
        rng = bundle["ranges"][f]
        is_integer = bool((df[f] == df[f].round()).all())
        info.append({
            "name": f,
            "min": rng["min"],
            "max": rng["max"],
            "mean": round(rng["mean"]) if is_integer else round(rng["mean"], 4),
            "step": 1 if is_integer else "any",
        })
    return info


@app.route("/")
def home():
    cards = []
    for key, cfg in DATASETS.items():
        m = get_bundle(key)["metrics"]
        cards.append({"key": key, "title": cfg["title"],
                      "description": cfg["description"], "r2": m["test_r2"]})
    return render_template("index.html", cards=cards)


@app.route("/predict/<name>", methods=["GET", "POST"])
def predict_page(name):
    if name not in DATASETS:
        abort(404)
    bundle = get_bundle(name)
    cfg = bundle["config"]
    features = feature_info(bundle)

    prediction = None
    entered = {f["name"]: f["mean"] for f in features}  # default values
    error = None

    if request.method == "POST":
        try:
            entered = {f["name"]: float(request.form[f["name"]]) for f in features}
            prediction = round(predict(name, entered), 2)
            # show whole numbers as 4 instead of 4.0 when the form is re-drawn
            for f in features:
                if f["step"] == 1:
                    entered[f["name"]] = int(entered[f["name"]])
        except (ValueError, KeyError):
            error = "Please enter a valid number in every field."

    return render_template(
        "predict.html", key=name, cfg=cfg, features=features,
        entered=entered, prediction=prediction, error=error,
        metrics=bundle["metrics"],
        coefs=list(zip(bundle["features"], bundle["model"].coef_)),
        intercept=bundle["model"].intercept_,
    )


@app.route("/figures/<path:filename>")
def figures(filename):
    return send_from_directory("figures", filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
