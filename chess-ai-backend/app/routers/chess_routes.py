from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import chess
import torch
from app.models.chess_model import ChessAI
import random  # Add import for random

router = APIRouter(prefix="/chess", tags=["chess"])
model = ChessAI()  # This will load our actor-critic models

class MoveRequest(BaseModel):
    fen: str

class GameData(BaseModel):
    pgn: str
    result: str

@router.post("/get-move")
async def get_ai_move(move_request: MoveRequest):
    try:
        print(f"Received FEN: {move_request.fen}")
        
        # Convert FEN to board state
        try:
            board = chess.Board(move_request.fen)
            print(f"Board state:\n{board}")
            print(f"Legal moves: {[move.uci() for move in board.legal_moves]}")
        except ValueError as e:
            print(f"Invalid FEN error: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid FEN string: {str(e)}")
        
        if board.is_game_over():
            print("Game is already over")
            raise HTTPException(status_code=400, detail="Game is already over")
        
        # Get move from the AI model
        try:
            move = model.get_move(board)
            print(f"AI selected move: {move}")
            
            # Validate the move
            move_uci = chess.Move.from_uci(f"{move['from']}{move['to']}")
            if move['promotion']:
                # Convert promotion piece symbol to piece type
                promotion_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
                promotion_type = promotion_map.get(move['promotion'])
                if promotion_type:
                    move_uci = chess.Move(
                        from_square=chess.parse_square(move['from']),
                        to_square=chess.parse_square(move['to']),
                        promotion=promotion_type
                    )
            
            # Check if move is legal
            if move_uci not in board.legal_moves:
                print(f"WARNING: AI generated illegal move: {move}")
                # Use a random legal move instead of the first one
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    random_move = random.choice(legal_moves)  # Use random choice instead of first move
                    move = {
                        'from': chess.square_name(random_move.from_square),
                        'to': chess.square_name(random_move.to_square),
                        'promotion': chess.piece_symbol(random_move.promotion) if random_move.promotion else None
                    }
                    print(f"Using random legal move: {move}")
            
            return {"move": move}
        except Exception as e:
            print(f"Error getting AI move: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error getting AI move: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/save-game")
async def save_game(game_data: GameData):
    try:
        # Save game data for model training
        model.save_game_data(game_data.pgn, game_data.result)
        return {"status": "success", "message": "Game data saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/model-info")
async def get_model_info():
    return {
        "status": "active",
        "model_version": model.version,
        "total_games_trained": model.total_games
    }
