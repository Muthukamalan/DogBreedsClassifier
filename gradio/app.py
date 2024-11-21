import os
import torch
import lightning as pl
import gradio as gr
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer
from torch.nn import functional as F

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_device(device=device)
torch.autocast(enabled=True, dtype='float16', device_type='cuda')

pl.seed_everything(123, workers=True)

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class_labels = [
    'Beagle',
    'Boxer',
    'Bulldog',
    'Dachshund',
    'German_Shepherd',
    'Golden_Retriever',
    'Labrador_Retriever',
    'Poodle',
    'Rottweiler',
    'Yorkshire_Terrier',
]


# Model
model = torch.jit.load('best_model.pt', map_location=device).to(device)


@torch.no_grad()
def predict_fn(img: Image):
    start_time = timer()
    try:
        # img = np.array(img)
        # print(img)
        img = TEST_TRANSFORMS(img).to(device)
        # print(type(img),img.shape)
        logits = model(img.unsqueeze(0))
        probabilities = F.softmax(logits, dim=-1)
        # print(torch.topk(probabilities,k=2))
        y_pred = probabilities.argmax(dim=-1).item()
        confidence = probabilities[0][y_pred].item()
        predicted_label = class_labels[y_pred]
        # print(confidence,predicted_label)
        pred_time = round(timer() - start_time, 5)
        res = {f'Title: {predicted_label}': confidence}
        return (res, pred_time)
    except Exception as e:
        print(f'error:: {e}')
        gr.Error('An error occured 💥!', duration=5)
        return ({'Title ☠️': 0.0}, 0.0)


gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type='pil'),
    outputs=[
        gr.Label(num_top_classes=1, label='Predictions'),  # what are the outputs?
        gr.Number(label='Prediction time (s)'),
    ],
    examples=[
        ['examples/' + i]
        for i in os.listdir(os.path.join(os.path.dirname(__file__), 'examples'))
    ],
    title='Dog Breeds Classifier 🐈',
    description='CNN-based Architecture for Fast and Accurate DogsBreed Classifier',
    article='Created by muthukamalan.m ❤️',
    cache_examples=True,
).launch(share=False, debug=False)
