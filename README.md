# 🩺 Skin Disease AI Assistant

A production-grade medical AI system for skin disease classification and LLM-powered clinical advice. This project combines Deep Learning (PyTorch) for disease classification with Large Language Models (Gemini/Groq/Ollama) to provide structured medical recommendations.


## 🚀 Features

- **Disease Detection:** Image classification using ResNet-50 / EfficientNet-B0 / Yolov8.
- **LLM Integration:** Real-time advice via Groq (Cloud) or Ollama (Local) LLM.
- **Patient Management:** Track history by Patient ID and Name.
- **Reports:** Automated PDF report generation with patient details, AI recommendations and download full report.
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
- Groq / Gemini API Key for using Cloud LLM
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

| File        | Used By        | Recommended Contents                                |
| :---------- | :------------- | :-------------------------------------------------- |
| `.env`      | Docker Compose | Variables for the container (API keys, Ollama URL). |
| `.env.dev`  | Local Python   | Local file paths, `ENV_MODE=dev`, dev API keys.     |
| `.env.prod` | App (Internal) | Production settings (usually mirrored from `.env`). |


Create a `.env.dev` file in the root directory. Paste the followings in that file:
```ini
APP_NAME=SkinCare_AI
ENV_MODE=dev
LLM_PROVIDER=Gemini

# LLM Provider: Groq
GROQ_API_KEY=groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# LLM Provider: Gemini
GOOGLE_API_KEY=gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# LLM Provider: Ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3-vl:2b

# Storage
DATABASE_URL=sqlite+aiosqlite:///./data/db/skin_app.db
UPLOAD_DIR=data/uploads

```

### 4. LLM Setup (Local/Cloud)

- ### Ollama

  If you prefer to run the LLM locally for privacy or offline use:

  1. **Install Ollama:** Download and install from [ollama.ai](https://ollama.ai/).
  2. **Pull the required model:**
      ```bash
      ollama pull qwen3-vl:2b
      ```
  3. **Configure Environment:** Ensure your `.env.dev` file has `LLM_PROVIDER=Ollama` and `OLLAMA_MODEL=qwen3-vl:2b`.
  
  4. **CORS/Docker Note:** If running the backend in Docker while Ollama is on the host, set the ollama based url as following:
  ```bash
  OLLAMA_BASE_URL=http://your_local_machine_ip:11434
  ```

- ### Gemini LLM
  If you prefer to use models from google cloud use following steps:  
  1. Use your google account and head to google developer api.
  2. Create an API Key from the `API Keys` tab.
  3. Put the api key in the `.env.dev` at `GOOGLE_API_KEY` and ensure the `LLM Provider=Gemini`.


- ### Groq Cloud LLM
  If you prefer to use models from cloud provides your can use groq.  
  1. Create an account at [Groq](https://console.groq.com/keys)
  2. Create an API Key from the `API Keys` tab.
  3. Put the api key in the `.env.dev` at `GROQ_API_KEY` and ensure the `LLM Provider=Groq`.


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
Access the frontend: `http://localhost:8501/`

## 🐳 Docker Deployment

The project is configured for easy deployment using Docker Compose. Before running the docker compose make sure to create `.env` in the root directory with all variables mentioned above. If using Ollama as LLM provides change the following:
```bash
OLLAMA_BASE_URL=http://your_local_machine_ip:11434
```

### CPU Deployment (Default)
```bash
rav run docker-compose
```

### GPU Deployment
1. Edit `docker-compose.yml` and change `dockerfile: Dockerfile` to `dockerfile: Dockerfile.gpu`.
2. Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed on your host.
3. Run:
   ```bash
   docker compose up --build -d
   ```
---

## 📂 API Documentation

Once the backend is running, you can access the interactive documentation:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### 🩺 Skin Analysis
#### `POST /api/v1/analyze_skin`
Analyzes an uploaded skin image and provides streaming LLM advice.

**Request:** `multipart/form-data`
- `user_id` (integer): Unique Patient ID.
- `patient_name` (string): Full name of the patient.
- `age` (integer): Patient age.
- `file` (binary): Image file (JPG, PNG).

**Response:** `text/event-stream`
The stream starts with a JSON metadata block followed by LLM tokens:
```json
{
  "user_id": "123",
  "patient_name": "John Doe",
  "age": 25,
  "prediction": "Eczema",
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
    "user_id": "123",
    "patient_name": "Imtiaz Ahammed",
    "age": 25,
    "image_path": "data/uploads/01.jpg",
    "prediction": "Eczema",
    "accuracy": 0.70,
    "llm_recommendation": "The diagnosis is Eczema...",
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
      "model": "qwen3-vl:2b"
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
    "efficientet.pt",
    "yolov8-cls.pt"
  ],
  "active_model": "resnet_v1.pt"
} 
```

#### `POST /api/v1/models/select`
Switches between the available pre-trained skin-disease model.  
**Query Parameter:**   `model_name=resnet.pt`  
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
3. `Yolov8n_cls`

### Training a new model:
```bash
python scripts/train.py --data_path "C:/path/to/dataset" --model_type resnet --epochs 20
```
**Outputs:** It contains model weights, training history plots, confusion matrix under the `scripts/output/<model_type>/`.

---

## 📁 Directory Structure
```markdown
├── api/             
│   └── v1/        # API endpoints
├── core/          # Central Config
│   ├── config.py       
│   ├── logger.py  # App logger
│   ├── db.py      # Db engine 
│   └── dependency.py   
├── models/  # Data Structures & Weights
│   ├── db_models.py   
│   ├── schemas.py # Pydantic models 
│   └── weights/   # .pt files 
├── services/           
│   ├── interface/ # AbstractClasses
│   ├── skin_service.py 
│   └── advisor_service.py
├── system_prompts/    
│   └── prompt_v1.py    
├── utils/            
│   ├── groq_client.py      
│   ├── ollama_client.py   
│   ├── gemini_client.py     
│   ├── file_validator.py
│   ├── visualization.py 
│   └── ui_helpers.py    
├── data/                   
│   ├── db/   # SQLite file 
│   ├── uploads/       
│   └── class_mapping.json 
├── scripts/                
│   └── train.py            # YOLO/CNN Training script
├── tests/            
│   └── test_api.py         # Pytest for API endpoints
├── logs/                  
│   └── app_date.log
├── ui.py        
├── main.py      
├── Dockerfile          
├── docker-compose.yml     
├── pyproject.toml   
├── README.md          
└── LICENSE
```


