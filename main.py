from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import io
import uvicorn
from typing import Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="REST API for detecting deepfake/AI-generated images using SigLIP model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and processor
model = None
processor = None

# Load the deepfake detection model
def load_model():
    global model, processor
    try:
        model_name = "prithivMLmods/deepfake-detector-model-v1"
        logger.info(f"Loading model: {model_name}")
        
        model = SiglipForImageClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Set model to evaluation mode
        model.eval()
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Deepfake detection endpoint
@app.post("/detect-deepfake")
async def detect_deepfake(file: UploadFile = File(...)) -> Dict:
    """
    Detect if an uploaded image is a deepfake/AI-generated image.
    
    Args:
        file: Uploaded image file (JPG/PNG)
        
    Returns:
        JSON response with detection results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image (JPG/PNG)"
            )
        
        # Read and validate image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        # Load and process image
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Preprocess image for model
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
        
        # Extract results
        # Class 0: fake, Class 1: real
        fake_prob = probabilities[0].item()
        real_prob = probabilities[1].item()
        
        # Determine if image is deepfake (threshold can be adjusted)
        is_deepfake = fake_prob > real_prob
        confidence_score = max(fake_prob, real_prob)
        
        response = {
            "is_deepfake": is_deepfake,
            "confidence_score": round(confidence_score, 4),
            "probabilities": {
                "fake": round(fake_prob, 4),
                "real": round(real_prob, 4)
            },
            "filename": file.filename,
            "file_size": len(contents)
        }
        
        logger.info(f"Processed image: {file.filename}, Result: {'Deepfake' if is_deepfake else 'Real'}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Root endpoint with API information
@app.get("/")
async def root():
    return {
        "message": "Deepfake Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect-deepfake": "Upload image to detect if it's a deepfake",
            "GET /health": "Check API health status",
            "GET /docs": "Interactive API documentation"
        },
        "supported_formats": ["JPG", "JPEG", "PNG"],
        "model": "prithivMLmods/deepfake-detector-model-v1"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)