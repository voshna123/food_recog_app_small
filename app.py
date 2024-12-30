
from foodrecog_demo import create_effnetB2_model


import gradio as gr


from timeit import default_timer as timer
from typing import Tuple, Dict
import torch 
import os

effnet_B_model, effnetB2_transforms = create_effnetB2_model()

class_names = ["pizza", "steak", "sushi"]


def pred(img)-> Tuple[Dict, float]:


  Results_dict = {}
  img = effnetB2_transforms(img).unsqueeze(dim = 0)

  effnet_B_model.load_state_dict(torch.load(f = "effnetB2_model_5_epochs_20%_data.pth"))

  model = effnet_B_model.eval()

  start_timer = timer()
  with torch.inference_mode():
    pred_logits = model(img)

  pred_probs = torch.softmax(pred_logits, dim = 1)

  pred_labels = torch.argmax(pred_probs, dim = 1)

  Results_dict[class_names[0]] = pred_probs[0][0]
  Results_dict[class_names[1]] = pred_probs[0][1]
  Results_dict[class_names[2]] = pred_probs[0][2]

  end_timer = timer()

  pred_time = round(end_timer - start_timer, 2)

  return Results_dict, pred_time



title = "Food Recognition"
description ="For pizza , steak and sushi"

demo = gr.Interface(fn = pred,
                    inputs = gr.Image(type = "pil"),
                    outputs = [gr.Label(num_top_classes = 3, label = "Predictions"),
                               gr.Number(label = "Prediction time (seconds)")],
                    title = title,
                    description = description,
                    )

demo.launch(debug = False)
