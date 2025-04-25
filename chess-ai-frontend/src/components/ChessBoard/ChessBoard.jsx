import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Chessboard } from 'react-chessboard';
import { Chess } from 'chess.js';
import './ChessBoard.css';

const ChessBoard = ({ onGameEnd }) => {
  const [game, setGame] = useState(() => new Chess());
  const [boardOrientation, setBoardOrientation] = useState('white');
  const [isAIThinking, setIsAIThinking] = useState(false);
  const [moveHistory, setMoveHistory] = useState([]);
  const [playerColor, setPlayerColor] = useState('white');
  const [selectedSquare, setSelectedSquare] = useState(null);
  const [legalMoveSquares, setLegalMoveSquares] = useState({});
  // Use a ref to track the current game state for async operations
  const gameRef = useRef(new Chess());

  // Update the ref whenever game state changes
  useEffect(() => {
    gameRef.current = new Chess(game.fen());
  }, [game]);

  function showLegalMoves(sourceSquare) {
    if (!sourceSquare) return [];
    
    const legalMoves = game.moves({
      square: sourceSquare,
      verbose: true
    });
    
    console.log(`Legal moves from ${sourceSquare}:`, legalMoves);
    return legalMoves;
  }

  // Make AI move - wrapped in useCallback to fix dependency warning
  const makeAIMove = useCallback(async () => {
    if (isAIThinking) {
      console.log("AI is already thinking, skipping move request");
      return;
    }

    setIsAIThinking(true);
    console.log("AI is thinking...");
    
    try {
      // Use the gameRef to get the current state
      const currentFEN = gameRef.current.fen();
      console.log("Current FEN for AI move:", currentFEN);

      const response = await fetch('http://localhost:8000/chess/get-move', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fen: currentFEN }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }

      const data = await response.json();
      console.log("AI selected move:", data.move);
      
      if (!data.move || !data.move.from || !data.move.to) {
        throw new Error("Invalid move data received from AI: " + JSON.stringify(data));
      }
      
      // Use the gameRef to make the AI move
      const gameCopy = new Chess(currentFEN);
      
      try {
        const moveObj = {
          from: data.move.from,
          to: data.move.to,
          promotion: data.move.promotion
        };
        
        console.log("AI move being applied:", moveObj);
        const result = gameCopy.move(moveObj);
        
        if (!result) {
          throw new Error("Failed to apply AI move: " + JSON.stringify(moveObj));
        }
        
        console.log("AI move successfully applied:", result);
        console.log("New board position:", gameCopy.fen());
        
        // Update the game state and ref
        setGame(gameCopy);
        gameRef.current = gameCopy;
        
        setMoveHistory(prev => [...prev, result.san]);
        
        if (gameCopy.isGameOver()) {
          handleGameOver(gameCopy);
        }
      } catch (moveError) {
        console.error("Error applying AI move:", moveError);
      }
    } catch (error) {
      console.error("Error during AI move:", error);
    } finally {
      setIsAIThinking(false);
    }
  }, [isAIThinking]);

  function onDrop(sourceSquare, targetSquare, piece) {
    const currentTurn = gameRef.current.turn() === 'w' ? 'white' : 'black';
    if (currentTurn !== playerColor) {
      console.log("Not your turn!");
      return false;
    }

    console.log("User attempting move from", sourceSquare, "to", targetSquare);
    
    const legalMoves = showLegalMoves(sourceSquare);
    const isLegalTarget = legalMoves.some(move => move.to === targetSquare);
    
    if (!isLegalTarget) {
      console.error(`${targetSquare} is not a legal target from ${sourceSquare}`);
      return false;
    }
    
    // If this is a promotion move, get the promotion piece from the piece parameter
    const promotion = piece ? piece.charAt(1).toLowerCase() : null;
    return makePlayerMove(sourceSquare, targetSquare, promotion);
  }

  function makePlayerMove(sourceSquare, targetSquare, promotion = null) {
    console.log("Attempting player move from", sourceSquare, "to", targetSquare);
    
    const gameCopy = new Chess(gameRef.current.fen());
    
    const legalMoves = gameCopy.moves({
      square: sourceSquare,
      verbose: true
    });
    
    const legalMove = legalMoves.find(move => move.to === targetSquare);
    if (!legalMove) {
      console.error(`${targetSquare} is not a legal target from ${sourceSquare}`);
      return false;
    }
    
    try {
      console.log("Move being applied:", legalMove);
      
      const result = gameCopy.move({
        from: legalMove.from,
        to: legalMove.to,
        promotion: promotion
      });
      
      if (!result) {
        console.error("Failed to apply legal move:", legalMove);
        return false;
      }
      
      console.log("Player move successfully applied:", result);
      console.log("New board position:", gameCopy.fen());
      
      // Update both state and ref
      setGame(gameCopy);
      gameRef.current = gameCopy;
      
      setMoveHistory(prev => [...prev, result.san]);
      
      if (gameCopy.isGameOver()) {
        handleGameOver(gameCopy);
        return true;
      }
      
      // Now make the AI move
      setTimeout(() => {
        makeAIMove();
      }, 300);
      
      return true;
    } catch (error) {
      console.error("Error making player move:", error);
      return false;
    }
  }

  // Check if it's AI's turn when the component mounts or game state changes
  useEffect(() => {
    if (!isAIThinking) {
      const timeoutId = setTimeout(() => {
        const currentTurn = gameRef.current.turn() === 'w' ? 'white' : 'black';
        
        if (currentTurn !== playerColor && !gameRef.current.isGameOver()) {
          makeAIMove();
        }
      }, 500);
      
      return () => clearTimeout(timeoutId);
    }
  }, [game, playerColor, isAIThinking, makeAIMove]);

  function handleGameOver(currentGame) {
    const gameToUse = currentGame || game;
    let result = 'draw';
    if (gameToUse.isCheckmate()) {
      result = gameToUse.turn() === 'w' ? 'black' : 'white';
    }

    console.log("Game over! Result:", result);
    saveGameData(result, gameToUse);
    
    if (onGameEnd) {
      onGameEnd({
        result,
        pgn: gameToUse.pgn(),
        history: moveHistory,
      });
    }
  }

  async function saveGameData(result, currentGame) {
    try {
      await fetch('http://localhost:8000/chess/save-game', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pgn: currentGame.pgn(),
          result: result,
        }),
      });
    } catch (error) {
      console.error('Error saving game:', error);
    }
  }

  function resetGame() {
    const newGame = new Chess();
    setGame(newGame);
    gameRef.current = newGame;
    setMoveHistory([]);
    setSelectedSquare(null);
    setLegalMoveSquares({});
    
    if (playerColor === 'black') {
      setTimeout(makeAIMove, 500);
    }
  }

  function flipBoard() {
    const newOrientation = boardOrientation === 'white' ? 'black' : 'white';
    setBoardOrientation(newOrientation);
    setPlayerColor(newOrientation);
    
    const newGame = new Chess();
    setGame(newGame);
    gameRef.current = newGame;
    setMoveHistory([]);
    setSelectedSquare(null);
    setLegalMoveSquares({});
    
    if (newOrientation === 'black') {
      setTimeout(makeAIMove, 500);
    }
  }

  function onSquareClick(square) {
    console.log("Square clicked:", square);
    
    const currentTurn = gameRef.current.turn() === 'w' ? 'white' : 'black';
    if (currentTurn !== playerColor) {
      console.log("Not your turn!");
      return;
    }
    
    if (!selectedSquare) {
      const piece = gameRef.current.get(square);
      if (piece) {
        const pieceColor = piece.color === 'w' ? 'white' : 'black';
        
        if (pieceColor === playerColor) {
          setSelectedSquare(square);
          
          const legalMoves = showLegalMoves(square);
          
          const legalSquares = {};
          legalMoves.forEach(move => {
            legalSquares[move.to] = {
              background: 'radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)',
              borderRadius: '50%'
            };
          });
          
          setLegalMoveSquares(legalSquares);
        }
      }
      return;
    }
    
    if (selectedSquare) {
      if (selectedSquare === square) {
        setSelectedSquare(null);
        setLegalMoveSquares({});
        return;
      }
      
      const legalMoves = showLegalMoves(selectedSquare);
      const isLegalTarget = legalMoves.some(move => move.to === square);
      
      if (!isLegalTarget) {
        const piece = gameRef.current.get(square);
        if (piece) {
          const pieceColor = piece.color === 'w' ? 'white' : 'black';
          if (pieceColor === playerColor) {
            setSelectedSquare(square);
            
            const newLegalMoves = showLegalMoves(square);
            const newLegalSquares = {};
            newLegalMoves.forEach(move => {
              newLegalSquares[move.to] = {
                background: 'radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)',
                borderRadius: '50%'
              };
            });
            
            setLegalMoveSquares(newLegalSquares);
            return;
          }
        }
        
        setSelectedSquare(null);
        setLegalMoveSquares({});
        return false;
      }
      
      const result = makePlayerMove(selectedSquare, square);
      
      setSelectedSquare(null);
      setLegalMoveSquares({});
      
      return result;
    }
  }

  function isDraggablePiece({ piece, sourceSquare }) {
    const pieceColor = piece.charAt(0).toLowerCase() === 'w' ? 'white' : 'black';
    const currentTurn = gameRef.current.turn() === 'w' ? 'white' : 'black';
    
    return currentTurn === playerColor && pieceColor === playerColor;
  }

  function getCustomSquareStyles() {
    const combinedStyles = { ...legalMoveSquares };
    
    if (selectedSquare) {
      combinedStyles[selectedSquare] = {
        backgroundColor: 'rgba(255, 255, 0, 0.4)',
      };
    }
    
    return combinedStyles;
  }

  return (
    <>
      <div className="chessboard-container">
        <div className="controls">
          <button onClick={resetGame}>New Game</button>
          <button onClick={flipBoard}>
            Play as {boardOrientation === 'white' ? 'Black' : 'White'}
          </button>
        </div>
        <div className="board">
          <Chessboard
            position={game.fen()}
            onPieceDrop={onDrop}
            boardOrientation={boardOrientation}
            customBoardStyle={{
              borderRadius: '5px',
              boxShadow: '0 5px 15px rgba(0, 0, 0, 0.5)',
            }}
            arePiecesDraggable={true}
            animationDuration={200}
            isDraggablePiece={isDraggablePiece}
            onSquareClick={onSquareClick}
            customSquareStyles={getCustomSquareStyles()}
            promotionToSquare={true}
          />
        </div>
        <div className="move-history">
          <h2>Move History</h2> 
          <ul>
            {moveHistory.map((move, index) => (
              <li key={index}>{move}</li>
            ))}
          </ul>
        </div>
        <div className="game-info">
          <div className="player-info">
            Playing as: <strong>{playerColor}</strong>
          </div>
          {isAIThinking && <div className="status">AI is thinking...</div>}
          {game.isGameOver() && (
            <div className="game-over">
              {game.isCheckmate()
                ? `Checkmate! ${game.turn() === 'w' ? 'Black' : 'White'} wins!`
                : 'Game ended in a draw'}
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ChessBoard;