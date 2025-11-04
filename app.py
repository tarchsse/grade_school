import pickle
import numpy as np
import pandas as pd
import gradio as gr

with open('rf_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
label_encoder = bundle.get('label_encoder', None)
problem_type = bundle.get('problem_type', 'classification')

feature_names = ['workload','quality','scrp']

def predict_with_prob(workload, quality, scrp):
    X = pd.DataFrame([[workload, quality, scrp]], columns=feature_names)
    pred = model.predict(X)[0]

    # ข้อความผลลัพธ์
    if problem_type == 'classification' and label_encoder is not None:
        pred_label = label_encoder.inverse_transform([pred])[0]
    else:
        pred_label = pred

    # ตาราง probability (ถ้ามี)
    prob_table = None
    if problem_type == 'classification' and hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        if label_encoder is not None:
            classes = label_encoder.inverse_transform(np.arange(len(probs)))
        else:
            classes = np.arange(len(probs))
        prob_table = pd.DataFrame({"class": classes, "probability": probs}).sort_values("probability", ascending=False)

    return str(pred_label), prob_table

inputs = [
    gr.Slider(minimum=0, maximum=100, step=1, value=25, label="Workload"),
    gr.Slider(minimum=0, maximum=40, step=1, value=10, label="Quality"),
    gr.Slider(minimum=0, maximum=10, step=1, value=2, label="SCRP"),
]

outputs = [
    gr.Textbox(label="Predicted Grade"),
    gr.Dataframe(label="Class Probabilities", wrap=True)
]

demo = gr.Interface(
    fn=predict_with_prob,
    inputs=inputs,
    outputs=outputs,
    title="Grade Prediction App by XGBoost",
    description="Adjust the workload, quality, scrp to predict grade and see probabilities (if available)"
)
demo.launch(share=True)
