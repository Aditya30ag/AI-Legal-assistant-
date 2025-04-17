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
# Use getenv without a default value to allow error handling
secret_key = os.getenv("GOOGLE_API_KEY")
if not secret_key:
    # For demonstration purposes only - should use environment variable in production
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

# Model configurations - Adjusted for more conversational responses
generation_config = {
    "temperature": 0.5,  # Increased for more natural responses
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2192,
}

# Initialize legal assistant model with updated system instructions for conversational responses
legal_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction='''You are LegalAssistAI, a friendly and approachable legal research assistant who speaks naturally while providing expert help.

Your capabilities include:
1. Analyzing legal documents and identifying key elements and potential issues
2. Providing accurate citations to relevant case law, statutes, and regulations
3. Offering preliminary legal analysis based on provided facts
4. Highlighting potential legal risks and considerations
5. Suggesting possible legal strategies based on precedent

When responding:
- Use conversational language and avoid excessive legal jargon
- Explain concepts in plain language before adding technical details
- Use contractions (like "you're" instead of "you are") when appropriate
- Vary sentence structure and length for natural rhythm
- Include brief acknowledgments of user concerns or questions
- Connect with users by using phrases like "I see your concern about..." or "I understand you're asking about..."

Important limitations - you must always:
1. Include a friendly disclaimer that your analysis is not legal advice
2. Avoid making definitive legal conclusions
3. Recommend consulting with a licensed attorney for specific legal advice
4. Be transparent about limitations in your knowledge
5. Refuse to draft complete legal documents but can provide templates or outlines
6. Maintain attorney-client privilege expectations by emphasizing confidentiality

Structure your responses to feel natural and conversational:
- Brief acknowledgment of the question
- Initial analysis in accessible language
- Relevant legal framework explained simply
- Potential considerations phrased conversationally
- Suggested next steps in a helpful tone
- Citations where relevant
- Friendly disclaimer

All answers should be firmly grounded in established legal principles and current law while sounding like a helpful colleague rather than a textbook.
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

def humanize_response(response_text):
    """Transform formal responses into more conversational language"""
    # Formal phrases to replace with more conversational alternatives
    replacements = {
        "It is important to note": "Keep in mind",
        "It is recommended that": "I'd recommend",
        "Please be advised": "Just so you know",
        "It is suggested that": "You might want to",
        "One must consider": "Consider",
        "It is necessary to": "You'll need to",
        "It is essential to": "It's essential to",
        "It is advised that": "I'd advise",
        "It is worth mentioning": "Worth mentioning",
        "It should be noted": "Note that",
        "In accordance with": "According to",
        "For the purpose of": "To",
        "In the event that": "If",
        "Prior to": "Before",
        "Subsequent to": "After",
        "In order to": "To",
        "Due to the fact that": "Because",
        "At this point in time": "Now",
        "At the present time": "Currently",
        "With regards to": "Regarding",
        "In reference to": "About",
    }
    
    for formal, casual in replacements.items():
        response_text = response_text.replace(formal, casual)
    
    return response_text

def create_conversational_closing():
    """Generate a varied conversational closing phrase"""
    closings = [
        "Hope that helps with your question!",
        "Does this address what you were looking for?",
        "Let me know if you need any clarification on this.",
        "Is there anything specific about this you'd like me to explain further?",
        "Would you like more information about any part of this?",
        "I'm here if you have follow-up questions about this.",
        "Does this give you what you needed?",
        "Hope this provides the guidance you were looking for.",
        "Let me know if you'd like me to explore any aspect of this further."
    ]
    return random.choice(closings)

def personalize_greeting(query):
    """Create a personalized greeting based on the legal query topic"""
    topic_keywords = {
        "contract": "I see you're asking about contract law",
        "divorce": "Regarding your question about divorce proceedings",
        "custody": "About your custody question",
        "property": "I understand you're inquiring about property matters",
        "employment": "Looking at your employment law question",
        "tenant": "Regarding your tenant/landlord question",
        "landlord": "About your landlord-tenant inquiry",
        "bankruptcy": "I see you're asking about bankruptcy issues",
        "injury": "Regarding your personal injury question",
        "accident": "About your accident-related inquiry",
        "damage": "I understand you're asking about damages",
        "will": "Regarding your question about wills",
        "trust": "About your question on trusts",
        "estate": "Regarding your estate planning question",
        "criminal": "I understand you're asking about criminal law",
        "discrimination": "About your discrimination concern",
        "harassment": "Regarding your harassment question"
    }
    
    default_greeting = "Thanks for your legal question"
    
    for keyword, greeting in topic_keywords.items():
        if keyword.lower() in query.lower():
            return greeting
    
    return default_greeting

class LegalProcessor:
    def __init__(self):
        self.legal_chat = legal_model.start_chat(history=[])
        self.lstm_processor = LSTMProcessor()
        self.conversation_context = {}  # Store user context for more personalized responses
        
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
            analysis_text = response.text
            
            # Create personalized greeting
            greeting = personalize_greeting(request.query)
            
            # Transform response to be more conversational
            humanized_response = f"{greeting}. {analysis_text}"
            humanized_response = humanize_response(humanized_response)
            
            # Add a conversational closing
            closing = create_conversational_closing()
            if not any(marker in humanized_response.lower() for marker in ["does this help", "hope this helps", "let me know if", "anything else"]):
                humanized_response += f"\n\n{closing}"
            
            # Create a more conversational disclaimer
            disclaimer_options = [
                "Just a friendly reminder: This is general information to help you understand the topic, not formal legal advice. For your specific situation, it's best to speak with an attorney.",
                "Remember, I can provide general guidance but this isn't a substitute for advice from a licensed attorney who knows the specifics of your situation.",
                "While I hope this information is helpful, please remember it's not legal advice. Consider talking to an attorney about your specific situation.",
                "I should mention that this information is meant to help you understand the legal concepts, but only a qualified attorney can give you proper legal advice for your specific circumstances."
            ]
            disclaimer = random.choice(disclaimer_options)
            
            # Extract citations if present (simplified extraction)
            citations = []
            
            result = {
                'analysis': humanized_response,
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