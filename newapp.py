from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI API
# Use getenv without a default value to allow error handling
secret_key = os.getenv("GOOGLE_API_KEY")
if not secret_key:
    # For demonstration purposes only - should use environment variable in production
    secret_key = "your_api_key_here"
    print("WARNING: Using placeholder API key. Set GOOGLE_API_KEY environment variable.")

# Configure the API
genai.configure(api_key=secret_key)

class LegalRequest(BaseModel):
    query: str
    case_details: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    jurisdiction: Optional[str] = "US Federal"
    use_lstm: Optional[bool] = False

class LegalResponse(BaseModel):
    analysis: str
    citations: Optional[list] = None
    disclaimer: str
    lstm_prediction: Optional[Dict[str, Any]] = None

class LSTMTrainingData(BaseModel):
    texts: List[str]
    labels: List[str]
    model_name: str = "legal_lstm_model"

class LSTMPredictionRequest(BaseModel):
    text: str
    model_name: str = "legal_lstm_model"

app = FastAPI(
    title="Legal AI Assistant API with LSTM",
    description="AI-powered solution addressing critical legal challenges through document analysis and guidance, enhanced with LSTM for sequence prediction",
    version="1.1.0"
)

# CORS Configuration
origins = [
    "http://localhost:3001",
    "http://localhost:5000",
    "*",  # Allow all origins - you should restrict this in production
]

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configurations
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Initialize legal assistant model with system instructions
legal_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction='''You are LegalAssistAI, an advanced legal research and analysis assistant designed to help legal professionals.

Your capabilities include:
1. Analyzing legal documents and identifying key elements and potential issues
2. Providing accurate citations to relevant case law, statutes, and regulations
3. Offering preliminary legal analysis based on provided facts
4. Highlighting potential legal risks and considerations
5. Suggesting possible legal strategies based on precedent

Important limitations - you must always:
1. Include a clear disclaimer that your analysis is not legal advice
2. Avoid making definitive legal conclusions
3. Recommend consulting with a licensed attorney for specific legal advice
4. Be transparent about limitations in your knowledge
5. Refuse to draft complete legal documents but can provide templates or outlines
6. Maintain attorney-client privilege expectations by emphasizing confidentiality

Structure your responses with: 
- Initial Analysis
- Relevant Legal Framework
- Potential Considerations
- Suggested Next Steps
- Citations
- Disclaimer

All answers should be firmly grounded in established legal principles and current law.
'''
)

class LSTMProcessor:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.label_encoders = {}
        self.model_dir = "lstm_models"
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models if available
        self._load_available_models()
    
    def _load_available_models(self):
        """Load any existing LSTM models from disk"""
        if os.path.exists(os.path.join(self.model_dir, "model_registry.json")):
            try:
                with open(os.path.join(self.model_dir, "model_registry.json"), "r") as f:
                    model_registry = json.load(f)
                    
                for model_name in model_registry:
                    self._load_model(model_name)
                print(f"Loaded {len(model_registry)} LSTM models")
            except Exception as e:
                print(f"Error loading model registry: {str(e)}")
    
    def _load_model(self, model_name):
        """Load a specific model and its associated tokenizer and label encoder"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.h5")
            tokenizer_path = os.path.join(self.model_dir, f"{model_name}_tokenizer.pkl")
            label_encoder_path = os.path.join(self.model_dir, f"{model_name}_labels.pkl")
            
            if os.path.exists(model_path) and os.path.exists(tokenizer_path) and os.path.exists(label_encoder_path):
                self.models[model_name] = load_model(model_path)
                
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizers[model_name] = pickle.load(f)
                
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoders[model_name] = pickle.load(f)
                
                print(f"Successfully loaded model: {model_name}")
                return True
            else:
                print(f"Could not find all required files for model: {model_name}")
                return False
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            return False
    
    def train_model(self, texts, labels, model_name="legal_lstm_model"):
        """Train an LSTM model with the provided text data and labels"""
        try:
            # Create and fit tokenizer
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(texts)
            
            # Convert texts to sequences
            sequences = tokenizer.texts_to_sequences(texts)
            max_seq_length = 100  # You can adjust this based on your data
            padded_sequences = pad_sequences(sequences, maxlen=max_seq_length)
            
            # Process labels
            unique_labels = sorted(list(set(labels)))
            label_encoder = {label: i for i, label in enumerate(unique_labels)}
            encoded_labels = np.array([label_encoder[label] for label in labels])
            
            # Create LSTM model
            model = Sequential()
            model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_seq_length))
            model.add(LSTM(128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(64))
            model.add(Dropout(0.2))
            model.add(Dense(len(unique_labels), activation='softmax'))
            
            model.compile(loss='sparse_categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
            
            # Train model
            model.fit(padded_sequences, encoded_labels, epochs=5, batch_size=32, validation_split=0.2)
            
            # Save model, tokenizer, and label encoder
            model_path = os.path.join(self.model_dir, f"{model_name}.h5")
            tokenizer_path = os.path.join(self.model_dir, f"{model_name}_tokenizer.pkl")
            label_encoder_path = os.path.join(self.model_dir, f"{model_name}_labels.pkl")
            
            model.save(model_path)
            
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
                
            with open(label_encoder_path, 'wb') as f:
                pickle.dump({v: k for k, v in label_encoder.items()}, f)  # Save inverted for prediction
            
            # Store in memory
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.label_encoders[model_name] = {v: k for k, v in label_encoder.items()}
            
            # Update model registry
            self._update_model_registry(model_name)
            
            return {
                "status": "success",
                "model_name": model_name,
                "num_examples": len(texts),
                "num_classes": len(unique_labels),
                "classes": unique_labels
            }
        
        except Exception as e:
            print(f"Error training LSTM model: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _update_model_registry(self, model_name):
        """Update the model registry with a new model"""
        registry_path = os.path.join(self.model_dir, "model_registry.json")
        
        if os.path.exists(registry_path):
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = []
            
        if model_name not in registry:
            registry.append(model_name)
            
        with open(registry_path, 'w') as f:
            json.dump(registry, f)
    
    def predict(self, text, model_name="legal_lstm_model"):
        """Make a prediction using the specified LSTM model"""
        if model_name not in self.models:
            success = self._load_model(model_name)
            if not success:
                return {
                    "status": "error",
                    "message": f"Model {model_name} not found"
                }
        
        try:
            # Prepare input text
            sequence = self.tokenizers[model_name].texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=100)  # Use same maxlen as training
            
            # Make prediction
            prediction = self.models[model_name].predict(padded_sequence)
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = self.label_encoders[model_name][predicted_class_index]
            
            # Get confidence scores for all classes
            confidence_scores = {}
            for i, score in enumerate(prediction[0]):
                class_name = self.label_encoders[model_name][i]
                confidence_scores[class_name] = float(score)
            
            return {
                "status": "success",
                "predicted_class": predicted_class,
                "confidence": float(prediction[0][predicted_class_index]),
                "all_scores": confidence_scores
            }
        
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def get_available_models(self):
        """Return a list of available trained models"""
        return list(self.models.keys())

class LegalProcessor:
    def __init__(self):
        self.legal_chat = legal_model.start_chat(history=[])
        self.lstm_processor = LSTMProcessor()
        
    def process_legal_query(self, request: LegalRequest) -> Dict[str, Any]:
        try:
            # Format the prompt to include all relevant information
            formatted_query = f"""
Jurisdiction: {request.jurisdiction}

Legal Query: {request.query}

"""
            # Add case details if provided
            if request.case_details:
                formatted_query += "Case Details:\n"
                for key, value in request.case_details.items():
                    formatted_query += f"- {key}: {value}\n"
            
            # Get legal analysis response
            response = self.legal_chat.send_message(formatted_query)
            
            # Extract citations if present (simplified extraction)
            citations = []
            analysis_text = response.text
            
            # Standard disclaimer
            disclaimer = "DISCLAIMER: This analysis is provided for informational purposes only and does not constitute legal advice. Please consult with a licensed attorney for advice concerning your specific situation."
            
            result = {
                'analysis': analysis_text,
                'citations': citations,
                'disclaimer': disclaimer
            }
            
            # Add LSTM prediction if requested
            if request.use_lstm:
                # Check if we have available models
                available_models = self.lstm_processor.get_available_models()
                if available_models:
                    # Use the first available model
                    lstm_prediction = self.lstm_processor.predict(request.query, model_name=available_models[0])
                    result['lstm_prediction'] = lstm_prediction
                else:
                    result['lstm_prediction'] = {
                        "status": "error", 
                        "message": "No LSTM models available. Please train a model first."
                    }
            
            return result
            
        except Exception as e:
            print(f"Error processing legal query: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize legal processor
legal_processor = LegalProcessor()

@app.post("/legal/analyze", response_model=LegalResponse)
async def analyze_legal_query(request: LegalRequest):
    """Process a legal query and return analysis with citations. 
    Optionally includes LSTM-based prediction if requested."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Legal query cannot be empty")
    
    try:
        result = legal_processor.process_legal_query(request)
        return LegalResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lstm/train")
async def train_lstm_model(data: LSTMTrainingData):
    """Train an LSTM model with provided text data and labels."""
    if len(data.texts) != len(data.labels):
        raise HTTPException(status_code=400, detail="Number of texts must match number of labels")
    
    if len(data.texts) < 10:
        raise HTTPException(status_code=400, detail="At least 10 examples required for training")
    
    try:
        result = legal_processor.lstm_processor.train_model(
            texts=data.texts, 
            labels=data.labels, 
            model_name=data.model_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lstm/predict")
async def predict_with_lstm(request: LSTMPredictionRequest):
    """Make a prediction using a trained LSTM model."""
    try:
        result = legal_processor.lstm_processor.predict(
            text=request.text,
            model_name=request.model_name
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lstm/models")
async def get_available_models():
    """Get a list of all available trained LSTM models."""
    try:
        models = legal_processor.lstm_processor.get_available_models()
        return {"available_models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Add a simple root endpoint for health checks."""
    return {
        "message": "Welcome to the Legal AI Assistant API with LSTM",
        "endpoints": {
            "POST /legal/analyze": "Analyze a legal query with optional case details and LSTM prediction",
            "POST /lstm/train": "Train a new LSTM model for legal text classification",
            "POST /lstm/predict": "Make predictions using a trained LSTM model",
            "GET /lstm/models": "Get a list of available trained LSTM models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)