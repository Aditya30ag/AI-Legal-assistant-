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
import random

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI API
secret_key = os.getenv("GOOGLE_API_KEY")
if not secret_key:
    secret_key = ""
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

# Model configurations - Adjusted for concise responses
generation_config = {
    "temperature": 0.2,  # Lower temperature for more focused responses
    "top_p": 0.85,
    "top_k": 40,
    "max_output_tokens": 800,  # Reduced token limit for shorter responses
}

# Initialize legal assistant model with improved system instructions for brief, concise responses
legal_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction='''You are LegalAssistAI, a brief and direct legal research assistant focused on providing concise information.

Your primary goal is BREVITY. Keep all responses under 4 paragraphs maximum, with no bullet points or numbered lists.

Your capabilities include:
- Analyzing legal documents to identify key issues
- Referencing relevant case law and regulations
- Offering preliminary legal analysis
- Highlighting potential legal considerations
- Suggesting possible strategies

Response guidelines - CRITICAL:
- Use 1-3 short paragraphs whenever possible
- NEVER use bullet points or numbered lists
- NEVER repeat information
- Present key information in order of importance
- Use plain, direct language with minimal legal jargon
- Focus on the most relevant points only
- Include only the most essential citations
- Use sentence fragments where appropriate for brevity
- Avoid phrases like "it's important to note" or "it should be mentioned"

Always include a one-sentence disclaimer at the end stating this is not legal advice.

Your responses should read like a brief memo from a colleague rather than a comprehensive report.
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

def create_brief_disclaimer():
    """Generate a concise legal disclaimer"""
    disclaimers = [
        "This is general information, not legal advice. Consult a lawyer for your specific situation.",
        "This information is educational only and not a substitute for legal counsel.",
        "For proper legal advice, please consult with a licensed attorney in your jurisdiction.",
        "This represents general guidance only; seek legal counsel for advice on your situation.",
        "This is not legal advice. Consult a qualified attorney in your jurisdiction."
    ]
    return random.choice(disclaimers)

def direct_response_format(text):
    """Format response to ensure brevity and directness"""
    # Remove bullet points and numbered lists
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Remove bullet points and numbering
        if line.strip().startswith('•') or line.strip().startswith('-') or line.strip().startswith('*'):
            formatted_lines.append(line.strip()[2:].strip())
        elif line.strip() and line.strip()[0].isdigit() and line.strip()[1:].startswith('. '):
            formatted_lines.append(line.strip()[3:].strip())
        else:
            formatted_lines.append(line)
    
    # Join lines that might have been in a list into a paragraph
    text = ' '.join([line for line in formatted_lines if line.strip()])
    
    # Limit to 4 paragraphs max
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 4:
        text = '\n\n'.join(paragraphs[:4])
    
    return text

class LegalProcessor:
    def __init__(self):
        self.legal_chat = legal_model.start_chat(history=[])
        self.lstm_processor = LSTMProcessor()
        self.conversation_context = {}
        
    def update_context(self, user_id, query):
        """Update conversation context for more personalized responses"""
        if not user_id:
            return
            
        if user_id not in self.conversation_context:
            self.conversation_context[user_id] = {"queries": [], "topics": set()}
        
        self.conversation_context[user_id]["queries"].append(query)
        
        # Extract potential topics from the query
        legal_topics = ["contract", "property", "divorce", "criminal", "employment", "landlord"]
        for topic in legal_topics:
            if topic in query.lower():
                self.conversation_context[user_id]["topics"].add(topic)
    
    def get_conversation_context(self, user_id):
        """Get relevant context from previous conversation if available"""
        if not user_id or user_id not in self.conversation_context:
            return None
            
        return self.conversation_context[user_id]
        
    def process_legal_query(self, request: LegalRequest) -> Dict[str, Any]:
        try:
            # Update user context if user_id provided
            if request.user_id:
                self.update_context(request.user_id, request.query)
            
            # Format the prompt with instruction for brevity
            formatted_query = f"""
            IMPORTANT: Provide a brief, direct response in 1-3 paragraphs using natural sentences. Do NOT use bullet points or numbered lists.
            
            Jurisdiction: {request.jurisdiction}
            Legal Query: {request.query}
            """
            
            # Add case details if provided
            if request.case_details:
                formatted_query += "Case Details:\n"
                for key, value in request.case_details.items():
                    formatted_query += f"{key}: {value}\n"
            
            # Get legal analysis response
            response = self.legal_chat.send_message(formatted_query)
            analysis_text = response.text
            
            # Ensure formatting is correct (no bullets, limited paragraphs)
            analysis_text = direct_response_format(analysis_text)
            
            # Create a concise disclaimer
            disclaimer = create_brief_disclaimer()
            
            # Extract citations if present (simplified extraction)
            citations = []
            
            result = {
                'analysis': analysis_text,
                'citations': citations,
                'disclaimer': disclaimer
            }
            
            # Add LSTM prediction if requested
            if request.use_lstm:
                available_models = self.lstm_processor.get_available_models()
                if available_models:
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
        "message": "Legal AI Assistant API with LSTM",
        "endpoints": {
            "POST /legal/analyze": "Analyze a legal query",
            "POST /lstm/train": "Train a new LSTM model",
            "POST /lstm/predict": "Make predictions with trained LSTM model",
            "GET /lstm/models": "Get available trained LSTM models"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)