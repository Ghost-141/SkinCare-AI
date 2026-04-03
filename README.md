# 🩺 Skin Disease AI Assistant

A production-grade medical AI system for skin disease classification and LLM-powered clinical advice. This project combines Deep Learning (PyTorch) for image analysis with Large Language Models (Groq/Ollama) to provide structured medical recommendations.

---

## 🚀 Features

- **Disease Detection:** Image classification using ResNet-50 / EfficientNet-B0.
- **LLM Integration:** Real-time advice via Groq (Cloud) or Ollama (Local) LLM.
- **Patient Management:** Track history by Patient ID and Name.
- **Reports:** Automated PDF report generation with patient details and AI recommendations.
- **Production Ready:** Async I/O, Environment-based configuration, and Dockerized deployment.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.13)
- **Frontend:** Streamlit
- **Database:** SQLAlchemy (SQLite for Development)
- **AI/ML:** PyTorch
- **LLM:** Groq API / Ollama (Local Llama 3.2:3b)
- **Deployment:** Docker & Docker Compose

---

## 📋 Project Setup

### 1. Prerequisites
- Python 3.13
- [Ollama](https://ollama.ai/) (Optional for local LLM)
- Groq API Key for using Cloud LLM
- Docker & Docker Compose (For containerized deployment)

### 2. Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ghost-141/SkinCare-AI.git
   cd SkinCare-AI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install uv
   uv sync
   ```

### 3. Environment Configuration (.env)
Create a `.env` file in the root directory. You can use `.env.dev` as a template:
```ini
APP_NAME=SkinDiseaseAI
ENV_MODE=dev

# LLM Provider: Groq or Ollama
LLM_PROVIDER=Groq
GROQ_API_KEY=your_gsk_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Ollama Settings (if using local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# Storage
DATABASE_URL=sqlite:///./data/db/skin_app.db
UPLOAD_DIR=data/uploads
```

### 4. LLM Setup 

- ### Ollama

  If you prefer to run the LLM locally for privacy or offline use:

  1. **Install Ollama:** Download and install from [ollama.ai](https://ollama.ai/).
  2. **Pull the required model:**
      ```bash
      ollama pull llama3.2:3b
      ```
  3. **Configure Environment:** Ensure your `.env` file has `LLM_PROVIDER=Ollama` and `OLLAMA_MODEL=llama3.2:3b`.
  4. **CORS/Docker Note:** If running the backend in Docker while Ollama is on the host, the `OLLAMA_BASE_URL` is automatically handled by the `docker-compose.yml` via `host.docker.internal`.

- ### Groq Cloud LLM
  If you prefer to use models from cloud provides your can use groq.  
  1. Create an account at [Groq](https://console.groq.com/keys)
  2. Create an API Key from the `API Keys` tab.
  3. Put the api key in the `.env` at `GROQ_API_KEY` and ensure the `LLM Provider=Groq`.


## Run the Project

- To run backend use the following commnad:
  ```bash
  rav run dev
  ```
- To run the frontend use the following command:
  ```bash
  rav run ui
  ```
Access the backend: `http://127.0.0.1:8000/docs`
Access the frontend: 

## 🐳 Docker Deployment

The project is configured for easy deployment using Docker Compose.

### CPU Deployment (Default)
```bash
docker-compose up --build -d
```

### GPU Deployment
1. Edit `docker-compose.yml` and change `dockerfile: Dockerfile` to `dockerfile: Dockerfile.gpu`.
2. Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your host.
3. Run:
   ```bash
   docker-compose up --build -d
   ```

**Note:** The Docker setup is pre-configured to connect to **Ollama running on your host machine** via `http://host.docker.internal:11434`.

---

## 📂 API Documentation

Once the backend is running, you can access the interactive documentation:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### 🩺 Skin Analysis
#### `POST /api/v1/analyze_skin`
Analyzes an uploaded skin image and provides streaming LLM advice.

**Request:** `multipart/form-data`
- `user_id` (string): Unique Patient ID.
- `patient_name` (string): Full name of the patient.
- `age` (integer): Patient age.
- `file` (binary): Image file (JPG, PNG).

**Response:** `text/event-stream`
The stream starts with a JSON metadata block followed by LLM tokens:
```json
{
  "user_id": "PATIENT-123",
  "patient_name": "John Doe",
  "age": 25,
  "prediction": "Acne",
  "accuracy": 0.95,
  "created_at": "2026-04-02T12:00:00Z"
}
```

### 📂 Patient History
#### `GET /api/v1/history/{user_id}`
Retrieves all past scan records for a specific patient.

**Response Body:**
```json
  {
    "id": 1,
    "user_id": "PATIENT-123",
    "patient_name": "Imtiaz Ahammed",
    "age": 25,
    "image_path": "data/uploads/01.jpg",
    "prediction": "Acne",
    "accuracy": 0.70,
    "llm_recommendation": "The diagnosis is acne...",
    "llm_provider": "Ollama",
    "created_at": "2026-04-02T12:00:00Z"
  }

```

### 🏥 System Health
#### `GET /api/v1/health`
Checks the operational status of all backend dependencies.

**Response Body:**
```json
{
  "status": "healthy",
  "services": {
    "database": "online",
    "skin_model": {
      "status": "loaded",
      "device": "cpu",
      "model_path": "models/weights/resnet.pt"
    },
    "llm": {
      "provider": "Ollama",
      "status": "online",
      "model": "llama-3.2:3B"
    }
  }
}
```

### 🤖 Model Management
#### `GET /api/v1/models`
Lists all available pre-trained model weights in the system.
**Response:** 
```json
{
  "available_models": [
    "resnet.pt",
    "efficientet.pt"
  ],
  "active_model": "resnet_v1.pt"
} 
```

#### `POST /api/v1/models/select`
Switches between the available pre-trained skin-disease model.
**Query Parameter:**   
`model_name=resnet.pt`
**Response:**
```json
{
  "message": "Active model successfully switched to resnet.pt",
  "status": "success",
  "active_model": "resnet.pt"
}
```

## 🧠 Model Training
The model was mainly trained on kaggle for better gpu support and longer training. However, with strong GPU we can train the model on our local environment. The scripts folder contains the training scripts to train, plot and save the models for deployment.
Currently, it supports training the following models:
1. `Resnet50`
2. `EfficientNet_b0`

### Training a new model:
```bash
python scripts/train.py --data_path "C:/path/to/dataset" --model_type resnet --epochs 20
```
**Outputs:** It contains model weights, training history plots, confusion matrix under the `scripts/output/<model_type>/`.

---

## 📁 Directory Structure
```markdown
├── app/             
│   ├── api/         
│   │   └── v1/      
│   ├── core/        
│   │   ├── config.py            
│   │   └── dependency.py       
│   ├── models/      
│   │   ├── db_models.py       
│   │   ├── schemas.py         
│   │   └── weights/            # .pt Model weight Files
│   ├── services/    
│   │   ├── interface/          # Abstract Base Classes (Interfaces)
│   │   ├── skin_service.py     # CNN Inference (TorchScript)
│   │   └── advisor_service.py  # LLM Logic (Groq/Ollama)
│   ├── system_prompts/         # Prompt templates
│   │   └── prompt_v1.py         
│   └── utils/       # Helper Modules & External Clients
│       ├── groq_client.py      # Groq Cloud API Wrapper
│       ├── ollama_client.py    # Ollama LLM Wrapper
│       ├── logger.py           # Logging Configuration
│       └── ui_helpers.py       # Streamlit UI Utilities
├── data/          
│   ├── db/                     # Local Database Files
│   ├── uploads/                # Temporary Storage for User Images
│   ├── class_mapping.json   # CNN 
Class Index -> Disease Name
│
├── scripts/         # Training Scripts
│   └── train.py                # Model Training Script
├── tests/           # Automated Testing
│   └── test_api.py             # Endpoint Integration Tests
├── logs/            # Application Runtime Logs
│   └── app.log
├── ui.py            # Streamlit Frontend
├── main.py          # FastAPI Entry Point
├── Dockerfile       # CPU Deployment
├── Dockerfile.gpu              
├── docker-compose.yml          
├── pyproject.toml              
├── README.md                   # Documentation
└── LICENSE                     # Project License
```


