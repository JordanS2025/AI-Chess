# Imports that are need 
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_chess
import random
import numpy as np
import chess
import chess.svg
from IPython.display import display, SVG
import os

# Actor Network Definition
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
# Critic Network Definition
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
#Move Handling Functions
def create_move_lookup():
    moves = []
    for from_square in range(64):
        for to_square in range(64):
            moves.append((from_square, to_square))
    return moves

def select_legal_action(action_probs, legal_moves):
    probs = action_probs.detach().numpy()[0]
    legal_moves_list = list(legal_moves)
    move_lookup = create_move_lookup()
    
    move_indices = []
    for move in legal_moves_list:
        from_square = move.from_square
        to_square = move.to_square
        try:
            idx = move_lookup.index((from_square, to_square))
            move_indices.append(idx)
        except ValueError:
            continue  # Skip if the move is not found in the lookup
    
    if not move_indices:
        # If no moves were matched, select a random legal move
        return random.choice(legal_moves_list)
    
    legal_probs = probs[move_indices]
    # Handle potential numerical issues
    legal_probs = np.clip(legal_probs, 1e-10, 1.0)
    if legal_probs.sum() == 0 or np.isnan(legal_probs.sum()):
        legal_probs = np.ones_like(legal_probs) / len(legal_probs)
    else:
        legal_probs = legal_probs / legal_probs.sum()
    
    selected_idx = np.random.choice(len(move_indices), p=legal_probs)
    return legal_moves_list[selected_idx]

def move_to_index(move, move_lookup):
    from_square = move.from_square
    to_square = move.to_square
    return move_lookup.index((from_square, to_square))


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)



def board_to_tensor(board):
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    state = np.zeros(768)
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_idx = pieces.index(piece.symbol())
            state[i + piece_idx * 64] = 1
            
    return torch.FloatTensor(state)


def calculate_reward(board, move, is_checkmate=False):
    base_reward = 0.0

    # Checkmate reward
    if is_checkmate:
        return 100.0  # Highest reward for winning

    # Piece values
    piece_values = {
        'p': 1.0,  # Pawn
        'n': 3.0,  # Knight
        'b': 3.0,  # Bishop
        'r': 5.0,  # Rook
        'q': 9.0,  # Queen
        'k': 0.0   # King (king captures are not applicable)
    }

    # Center control rewards
    center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}
    if move.to_square in center_squares:
        base_reward += 0.5

    # Piece development rewards
    from_piece = board.piece_at(move.from_square)
    if from_piece and from_piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
        if move.from_square in [chess.B1, chess.G1, chess.B8, chess.G8]:  # Starting squares
            base_reward += 0.3

    # King safety (castling)
    if from_piece and from_piece.piece_type == chess.KING:
        if chess.square_distance(move.from_square, move.to_square) > 1:
            base_reward += 1.0

    # Capture rewards
    captured_piece = board.piece_at(move.to_square)
    if captured_piece:
        base_reward += piece_values[captured_piece.symbol().lower()]

    return base_reward


def train_chess_ai(num_episodes=100, save_path='chess_model', batch_size=64, gamma=0.99):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    env = gym.make('Chess-v0')
    env.reset()
    chess_board = chess.Board()  # Create a new chess board instance

    actor_net = Actor(input_size=768, output_size=4672)
    critic_net = Critic(input_size=768)

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=1e-3)

    memory = ReplayBuffer(capacity=10000)
    training_history = []
    move_lookup = create_move_lookup()

    wins = 0
    games_played = 0

    def update_networks(states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_values = critic_net(next_states).squeeze(1)
            target_values = rewards + gamma * next_values * (1 - dones)

        current_values = critic_net(states).squeeze(1)
        critic_loss = F.mse_loss(current_values, target_values)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        action_probs = actor_net(states)
        advantages = (target_values - current_values).detach()

        action_log_probs = torch.log(action_probs + 1e-10)
        selected_action_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        actor_loss = -(selected_action_log_probs * advantages).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()

    for episode in range(num_episodes):
        env.reset()
        chess_board = chess.Board()  # Create fresh board for new episode
        done = False
        total_reward = 0
        episode_data = []

        while not done:
            state_tensor = board_to_tensor(chess_board).unsqueeze(0)
            action_probs = actor_net(state_tensor)
            legal_moves = list(chess_board.legal_moves)

            if not legal_moves:
                break  # No legal moves, end the game

            action = select_legal_action(action_probs, legal_moves)
            action_idx = move_to_index(action, move_lookup)

            # Perform the move
            observation, reward_env, done, info = env.step(action)
            chess_board = observation  # Direct assignment since observation is already a chess.Board
            # Check for checkmate
            is_checkmate = chess_board.is_checkmate()

            if is_checkmate:
                wins += 1
            games_played += 1

            # Calculate enhanced reward
            reward = calculate_reward(
                board=chess_board,
                move=action,
                is_checkmate=is_checkmate
            )

            next_state_tensor = board_to_tensor(chess_board).unsqueeze(0)
            memory.push(state_tensor, action_idx, reward, next_state_tensor, done)

            if len(memory) > batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states = torch.cat(states)
                next_states = torch.cat(next_states)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                critic_loss, actor_loss = update_networks(
                    states, actions, rewards, next_states, dones
                )

                episode_data.append({
                    'critic_loss': critic_loss,
                    'actor_loss': actor_loss
                })

            total_reward += reward

        training_history.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'moves': episode_data
        })

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        if episode % 10 == 0:
            win_rate = (wins / games_played) * 100
            print(f"Games played: {games_played}")
            print(f"Wins: {wins}")
            print(f"Win rate: {win_rate:.2f}%")

        if (episode + 1) % 100 == 0:
            torch.save(actor_net.state_dict(), f'{save_path}_actor.pth')
            torch.save(critic_net.state_dict(), f'{save_path}_critic.pth')
            np.save(f'{save_path}_history.npy', training_history)

    return actor_net, critic_net, training_history


# Model Initialization and Testing
input_size = 768  # Example input size (board state as a flat vector)
output_size = 4672  # Example output size (number of possible moves in chess)
actor_net = Actor(input_size=input_size, output_size=output_size)
critic_net = Critic(input_size=input_size)

# Run Training
# train_chess_ai(save_path='models/chess_ai')
actor_net, critic_net, history = train_chess_ai(
    num_episodes=1000,
    save_path='models/chess_ai',
    batch_size=64,
    gamma=0.99
)