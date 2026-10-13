"""
gradio_app.py
-------------
Gradio user interface for the two Linear Regression models.

Run:  python gradio_app.py      ->  http://127.0.0.1:7860
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from regression_core import DATASETS, get_bundle, predict


def make_slider(feature, rng, series):
    """Create a slider whose range comes from the real data."""
    is_integer = bool((series == series.round()).all())
    if is_integer:
        lo, hi, step = rng["min"], rng["max"], 1
        default = round(rng["mean"])
    else:
        # round the bounds outwards so the labels stay short and readable
        lo = math.floor(rng["min"] * 1000) / 1000
        hi = math.ceil(rng["max"] * 1000) / 1000
        step = 0.001
        default = round(rng["mean"], 3)
    return gr.Slider(
        minimum=lo, maximum=hi, step=step,
        value=default, label=feature,
    )


def metrics_markdown(bundle):
    m = bundle["metrics"]
    return (
        f"**Model performance on the test set**\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| R² (train) | {m['train_r2']:.4f} |\n"
        f"| R² (test) | {m['test_r2']:.4f} |\n"
        f"| MAE | {m['mae']:.4f} |\n"
        f"| MSE | {m['mse']:.4f} |\n"
        f"| RMSE | {m['rmse']:.4f} |"
    )


def build_tab(name):
    bundle = get_bundle(name)
    cfg = bundle["config"]
    df = bundle["df"]
    target = cfg["target"]

    gr.Markdown(f"### {cfg['title']}\n{cfg['description']}")

    with gr.Row():
        with gr.Column(scale=1):
            sliders = [
                make_slider(f, bundle["ranges"][f], df[f])
                for f in bundle["features"]
            ]
            btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            result = gr.Number(label=f"Predicted: {cfg['target_label']}",
                               precision=2)
            plot = gr.Plot(label="Where this prediction falls")
            gr.Markdown(metrics_markdown(bundle))

    def run(*values):
        pred = predict(name, list(values))
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.hist(df[target], bins=25, color="#9bc4e2", edgecolor="white")
        ax.axvline(pred, color="#d62828", linewidth=2.5,
                   label=f"Prediction: {pred:.1f}")
        ax.set_xlabel(target)
        ax.set_ylabel("Number of samples")
        ax.legend()
        fig.tight_layout()
        return round(pred, 2), fig

    btn.click(run, inputs=sliders, outputs=[result, plot])


with gr.Blocks(title="Linear Regression Lab") as demo:
    gr.Markdown(
        "# Linear Regression Predictor\n"
        "Move the sliders and press **Predict**. Each tab uses a different "
        "dataset and its own trained Linear Regression model."
    )
    with gr.Tabs():
        for key, cfg in DATASETS.items():
            with gr.Tab(cfg["title"]):
                build_tab(key)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
