from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import shutil
from app.model.model import make_inference
from app.model.model import __version__ as model_version


# Label classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

app = FastAPI()

class ImageRequest(BaseModel):
    file: UploadFile

# to check if it's working
@app.get('/')
def home():
    return {'check':'OK', 'model_version':model_version}



# upload photo and predict
@app.post("/upload-image/")
async def upload_image(image_request: ImageRequest):

    # Save the uploaded file locally
    uploaded_file_path = f"uploaded_images/{image_request.file.filename}"
    with open(uploaded_file_path, "wb") as image_file:
        shutil.copyfileobj(image_request.file.file, image_file)

    # Use the saved file path for inference
    label, confidence = make_inference(uploaded_file_path)
    result = classes[label]

    # Return the result or any other response you want
    return JSONResponse(content={"result": result, 'confidence': confidence, "model_version": model_version})