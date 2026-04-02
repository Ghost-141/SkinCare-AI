# 🩺 Skin Disease AI Assistant

A production-grade medical AI system for skin disease classification and LLM-powered clinical advice. This project combines Deep Learning (PyTorch) for image analysis with Large Language Models (Groq/Ollama) to provide structured medical recommendations.

---

## 🚀 Features

- **AI Analysis:** Image classification using ResNet-50 / EfficientNet-B0.
- **LLM Integration:** Real-time streaming advice via Groq (Cloud) or Ollama (Local).
- **Patient Management:** Track history by Patient ID and Name.
- **Advanced Health Monitoring:** Real-time status of DB, Disk, and LLM providers.
- **Reports:** Automated PDF report generation with patient details and AI findings.
- **Production Ready:** Async I/O, Environment-based configuration, and Dockerized deployment.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI (Python 3.11)
- **Frontend:** Streamlit
- **Database:** SQLAlchemy (SQLite for Dev, easily portable to PostgreSQL)
- **AI/ML:** PyTorch, TorchScript
- **LLM:** Groq API / Ollama (Local Llama 3)
- **Deployment:** Docker & Docker Compose

---

## 📋 Project Setup

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) (Optional, for local LLM)
- Docker & Docker Compose (For containerized deployment)

### 2. Local Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd skin_disease
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 3. Environment Configuration (.env)
Create a `.env` file in the root directory. You can use `.env.dev` as a template:
```ini
APP_NAME=SkinDiseaseAI
ENV_MODE=dev

# LLM Provider: Groq or Ollama
LLM_PROVIDER=Groq
GROQ_API_KEY=your_gsk_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama Settings (if using local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Storage
DATABASE_URL=sqlite:///./data/db/skin_app.db
UPLOAD_DIR=data/uploads
```

---

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

### Key Endpoints:
- `POST /api/v1/analyze_skin`: Upload image and get streaming LLM advice.
- `GET /api/v1/history/{user_id}`: Retrieve scan history for a specific patient ID.
- `GET /api/v1/health`: Check system status (DB, Disk, LLM Connectivity).

---

## 🧠 Model Training

We provide a dedicated script for training new models based on your research.

### Training a new model:
```bash
python scripts/train.py --data_path "C:/path/to/dataset" --model_type resnet --epochs 20
```
- **Outputs:** Models, class mappings, training history plots, and confusion matrices are saved in `scripts/output/<model_type>/`.

---

## 📁 Directory Structure
```text
├── api/             # FastAPI routes & endpoints
├── core/            # Configuration & Database logic
├── data/            # Local DB and Image uploads (Ignored by Git)
├── models/          # DB Models, Pydantic Schemas, and AI Weights
├── scripts/         # Training scripts and output visualizations
├── services/        # Business logic (AI Prediction, LLM Advisor)
├── utils/           # Shared utilities (Logging, Clients, UI helpers)
└── ui.py            # Streamlit Frontend
```

---

## 🛡️ License
This project is for educational and research purposes. Please consult a medical professional for actual diagnosis.
