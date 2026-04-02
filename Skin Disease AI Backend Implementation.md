This technical implementation guide is designed for an AI Engineer to build the full backend for the **Skin Disease Detection & LLM Advisor System**. It follows your specific directory structure and integrates the dual-LLM provider requirement along with SQLite persistence.

# ---

**🛠 AI Implementation Brief: Skin Disease Detection Backend**

## **1\. Project Overview & Objective**

Build a FastAPI-based modular backend that processes skin images via a computer vision pipeline and provides medical recommendations using either a local LLM (Ollama) or a cloud API (Groq).

## **2\. Core Technical Stack**

* **Framework:** FastAPI.

* **Database:** SQLite (SQLAlchemy ORM).

* **Vision Models:** PyTorch (ResNet/EfficientNet) or YOLOv8.

* **LLM Providers:**  
  * **Local:** Ollama (Model: llama3).  
  * **Cloud:** Groq (Model: llama-3.3-70b-versatile).  
* **Deployment:** Docker.

## ---

**3\. Database Schema (models/db\_models.py)**

The system must log every request to the data/db/skin\_app.db file with the following fields:

| Field | Type | Description |
| :---- | :---- | :---- |
| id | Integer | Primary Key.  |
| image\_path | String | Local path to the uploaded file. |
| prediction | String | Detected disease name (e.g., "Eczema").  |
| accuracy | Float | Model confidence score (0.0 to 1.0).  |
| llm\_recommendation | Text | Full markdown string of LLM advice.  |
| llm\_provider | String | 'Ollama' or 'Groq'. |
| created\_at | DateTime | Timestamp of the analysis.  |

## ---

**4\. Modular Implementation Logic**

### **A. API Versioning (api/v1/)**

All routes must be prefixed with /api/v1.

* **Route:** POST /analyze\_skin.

* **Logic:** 1\. Save image to data/uploads/. 2\. Call SkinService for classification. 3\. Call AdvisorService (using the provider toggled in config.py). 4\. Save results to SQLite. 5\. Return JSON response .

### **B. LLM Provider Strategy (utils/)**

Implement a factory pattern in AdvisorService to switch between:

1. **Groq Client:** Using the groq Python SDK for fast cloud inference.  
2. **Ollama Client:** Using the ollama Python library or a local requests call to localhost:11434.

### **C. Model Weight Management (models/weights/)**

The SkinService must load weights from this directory.

* Support for .pt (PyTorch) or .onnx formats.

* Ensure the preprocessing script in utils/ matches the training image transforms (Resize, Normalize).

### **C. Environment Setup**

Based on env file select llm provider, pre-trained model