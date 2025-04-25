import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
import json
from datetime import datetime
from pathlib import Path
import random


class ActorNetwork(nn.Module):
    def __init__(self, output_size=4672):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ChessAI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.version = "1.1.0"  # Updated version number
        self.total_games = 0
        self._load_models()
        self._load_stats()
        
    def _load_models(self):
        try:
            model_path = Path("/Users/bignola/Desktop/School/Capstone/AI-Chess-main/AI-Chess-main/ChampionModel")
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found at {model_path}")
            
            actor_path = model_path / "champion_actor.pth"
            critic_path = model_path / "champion_critic.pth"
            
            if not actor_path.exists():
                raise FileNotFoundError(f"Actor model not found at {actor_path}")
            if not critic_path.exists():
                raise FileNotFoundError(f"Critic model not found at {critic_path}")
            
            # Initialize new CNN models
            self.actor = ActorNetwork().to(self.device)
            self.critic = CriticNetwork().to(self.device)
            
            # Try to load the state dicts
            try:
                actor_state_dict = torch.load(actor_path, map_location=self.device)
                critic_state_dict = torch.load(critic_path, map_location=self.device)
                
                # Load state dicts
                self.actor.load_state_dict(actor_state_dict)
                self.critic.load_state_dict(critic_state_dict)
            except RuntimeError as e:
                print(f"Warning: Could not load existing model weights: {e}")
                print("Initializing with new weights instead")
            
            self.actor.eval()
            self.critic.eval()
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def _load_stats(self):
        stats_path = Path(__file__).parent.parent.parent / "models" / "stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.total_games = stats.get('total_games', 0)
        else:
            self._save_stats()

    def _save_stats(self):
        stats_path = Path(__file__).parent.parent.parent / "models" / "stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump({
                'total_games': self.total_games,
                'last_updated': datetime.now().isoformat()
            }, f)
    
    def _board_to_tensor(self, board):
        """
        Convert a chess board to a tensor suitable for CNN input.
        Returns a tensor of shape (1, 12, 8, 8) where:
        - 12 channels represent 6 piece types (pawn, knight, bishop, rook, queen, king) for both colors
        - 8x8 represents the board squares
        """
        # Initialize empty tensor
        board_tensor = torch.zeros((1, 12, 8, 8), device=self.device)
        
        # Map piece types to channel indices
        piece_to_channel = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        
        # Fill the tensor
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Calculate row and column
                row = 7 - (square // 8)  # Convert to 0-7 from top
                col = square % 8
                # Set the appropriate channel and position
                channel = piece_to_channel[piece.piece_type] + (6 if piece.color == chess.WHITE else 0)
                board_tensor[0, channel, row, col] = 1.0
        
        return board_tensor

    def get_move(self, board):
        with torch.no_grad():
            try:
                board_tensor = self._board_to_tensor(board)
                move_probs = self.actor(board_tensor)
                move_probs = torch.softmax(move_probs, dim=1)
                
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    raise ValueError("No legal moves available")
                
                move_indices = []
                for move in legal_moves:
                    from_square = move.from_square
                    to_square = move.to_square
                    index = from_square * 64 + to_square
                    if move.promotion:
                        index = 4096 + (move.promotion - 2) * 64 * 64 + from_square * 64 + to_square
                    move_indices.append(index)
                
                legal_move_probs = move_probs[0][move_indices]
                legal_move_probs = legal_move_probs / legal_move_probs.sum()
                
                selected_idx = torch.multinomial(legal_move_probs, 1)[0].item()
                move = legal_moves[selected_idx]
                
                result = {
                    'from': chess.square_name(move.from_square),
                    'to': chess.square_name(move.to_square),
                    'promotion': chess.piece_symbol(move.promotion) if move.promotion else None
                }
                
                return result
                
            except Exception as e:
                print(f"Error in get_move: {str(e)}")
                try:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        random_move = random.choice(legal_moves)
                        return {
                            'from': chess.square_name(random_move.from_square),
                            'to': chess.square_name(random_move.to_square),
                            'promotion': chess.piece_symbol(random_move.promotion) if random_move.promotion else None
                        }
                except Exception as fallback_error:
                    print(f"Error in fallback move selection: {str(fallback_error)}")
                raise

    def save_game_data(self, pgn, result):
        try:
            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create the data directory if it doesn't exist
            data_dir = Path(__file__).parent.parent.parent / "data" / "games"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create the game data file
            game_file = data_dir / f"game_{timestamp}.json"
            
            # Save the game data
            with open(game_file, 'w') as f:
                json.dump({
                    "pgn": pgn,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }, f)
            
            # Update total games count
            self.total_games += 1
            self._save_stats()
            
            return True
        except Exception as e:
            print(f"Error saving game data: {str(e)}")
            raise
