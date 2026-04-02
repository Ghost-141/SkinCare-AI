.

├── api/

│   └── v1/                        # Versioned API for future-proofing

│       ├── endpoints/

│       │   ├── skin\_analysis.py   # POST /analyze\_skin

│       │   └── chat.py            # LLM Chat interface

│       └── router.py              # APIRouter orchestration

├── core/

│   ├── config.py                  # Environment variables \& App settings

│   └── dependency.py              # Model \& Database injection

├── data/

│   ├── db/

│   │   └── skin\_app.db            # SQLite database for user logs

│   └── product\_info.json

├── models/

│   ├── schemas.py                 # Pydantic request/response models

│   ├── db\_models.py               # SQLite/SQLAlchemy table definitions

│   └── weights/                   # Directory for model checkpoints

│       ├── resnet\_v1.pt           # CNN weights

│       └── yolov8\_best.pt         # YOLO weights

├── services/

│   ├── interface/                 # Abstract base classes

│   │   ├── analysis\_interface.py

│   │   └── chat\_interface.py

│   ├── skin\_service.py            # Image preprocessing \& classification logic

│   └── advisor\_service.py         # LLM reasoning \& recommendation logic

├── system\_prompts/

│   └── prompt\_v1.py               # Versioned LLM system prompts

├── utils/

│   └── groq\_client.py             # LLM API client

├── Dockerfile                     # Deployment configuration

├── requirements.txt               # Project dependencies

└── main.py                        # FastAPI entry point

