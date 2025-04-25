# Chess AI Backend

A FastAPI-based backend for the Chess AI platform that handles model inference and game data collection.

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your model files:

- Put `actor.pth` and `critic.pth` in the `models/` directory

## Running the Server

Start the FastAPI server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `GET /`: Health check endpoint
- `POST /chess/get-move`: Get AI move for current board position
- `POST /chess/save-game`: Save completed game data
- `GET /chess/model-info`: Get information about the loaded model

## Model Integration

The backend expects PyTorch model files:

- `actor.pth`: The policy network that selects moves
- `critic.pth`: The value network that evaluates positions

Game data is saved in the `data/games` directory for future training.
