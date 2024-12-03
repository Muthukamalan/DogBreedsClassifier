from fastapi  import FastAPI,Depends,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

import contextlib
import numpy as np 
import onnx 
import onnxruntime as ort 


class_labels = [
    "Beagle",
    "Boxer",
    "Bulldog",
    "Dachshund",
    "German_Shepherd",
    "Golden_Retriever",
    "Labrador_Retriever",
    "Poodle",
    "Rottweiler",
    "Yorkshire_Terrier",
]


INPUT_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")                            # Convert to RGB if not already
    image = image.resize(INPUT_SIZE)                        # Resize
    img_array = np.array(image).astype(np.float32) / 255.0  # Convert to numpy array and normalize
    img_array = (img_array - MEAN) / STD                    # Apply mean and std normalization
    img_array = img_array.transpose(2, 0, 1)                # Transpose to channel-first format (NCHW)
    img_array = np.expand_dims(img_array, 0)                # Add batch dimension
    return img_array



class Dogsprediction:
    def load_model(self)->None:
        self.session = ort.InferenceSession('mambaout.onnx')
        self.session_input_name = self.session.get_inputs()[0].name
    
    def predict(self,image:Image.Image):
        img = preprocess_image(image=image)
        outputs = self.session.run(None,{self.session_input_name:img.astype(np.float32)})
        logits = outputs[0][0]
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        # predictions = {class_labels[i]: float(prob) for i, prob in enumerate(probabilities)}
        predicted_label = class_labels[np.argmax(probabilities)]
        confidence = np.max(probabilities)
        return {
            'confidence': float(confidence),
            'label':str(predicted_label)
        }
    
dog_prediction = Dogsprediction()

@contextlib.asynccontextmanager
async def lifespan(app:FastAPI):
    dog_prediction.load_model()
    yield

app = FastAPI(title="Image Classification API",lifespan=lifespan,description="FastAPI application serving an ONNX model for image classification",    version="1.0.0",)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add these lines after creating your FastAPI app
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def home():
    return FileResponse('static/index.html')

@app.post("/dogs",response_model=dict)
async def post_object_detection(image:UploadFile=File(...))->dict:
    image_objext = Image.open(image.file).convert("RGB")
    return dog_prediction.predict(image_objext)


@app.get("/health")
async def health_check():
    return JSONResponse(
        content={"status": "healthy", "model_loaded": True}, status_code=200
    )

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port='8080')