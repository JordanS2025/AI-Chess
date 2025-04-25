from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chess_routes

app = FastAPI(title="Chess AI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chess_routes.router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Chess AI API is running"}