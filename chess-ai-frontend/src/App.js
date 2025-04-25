import React, { useState, useEffect } from 'react';
import ChessBoard from './components/ChessBoard/ChessBoard';
import { checkAPIStatus } from './services/apiService';
import './App.css';
import logo from '/Users/bignola/Desktop/School/Capstone/AI-Chess-main/chess-ai-frontend/src/DeepMagnus2.0.png'
function App() {
  const [apiConnected, setApiConnected] = useState(false);
  const [gameResults, setGameResults] = useState(null);
  
  // Check if API is available on component mount
  useEffect(() => {
    const checkConnection = async () => {
      const isConnected = await checkAPIStatus();
      setApiConnected(isConnected);
    };
    
    checkConnection();
    // Set up interval to periodically check connection
    const interval = setInterval(checkConnection, 30000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Handle game end
  const handleGameEnd = (data) => {
    setGameResults(data);
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="logo" alt="Deep Magnus" />
        {!apiConnected && (
          <div className="connection-warning">
            Backend API not connected. Please start the FastAPI server.
          </div>
        )}
      </header>
      
      <main>
        <ChessBoard onGameEnd={handleGameEnd} />
        
        {gameResults && (
          <div className="game-results">
            <h2>Game Results</h2>
            <p><strong>Winner:</strong> {gameResults.result}</p>
            <p><strong>Total Moves:</strong> {gameResults.history.length}</p>
            <div className="move-history">
              <h3>Move History PGN</h3>
              <div className="moves">
                {gameResults.history.map((move, index) => (
                  <span key={index}>
                    {index % 2 === 0 ? `${Math.floor(index/2) + 1}. ` : ''}
                    {move} {index % 2 === 1 ? ' ' : ''}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
      
      <footer>
        <p>Created with React and FastAPI</p>
      </footer>
    </div>
  );
}
export default App;