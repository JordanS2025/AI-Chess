// Base URL for API
const API_BASE_URL = 'http://localhost:8000';

// Get AI move
export const getAIMove = async (fen) => {
  try {
    const response = await fetch(`${API_BASE_URL}/get-move`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ fen }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting AI move:', error);
    throw error;
  }
};

// Save game data
export const saveGame = async (pgn, result) => {
  try {
    const response = await fetch(`${API_BASE_URL}/save-game`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ pgn, result }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error saving game:', error);
    throw error;
  }
};

// Check if the API is available
export const checkAPIStatus = async () => {
  try {
    const response = await fetch(API_BASE_URL);
    return response.ok;
  } catch (error) {
    return false;
  }
};