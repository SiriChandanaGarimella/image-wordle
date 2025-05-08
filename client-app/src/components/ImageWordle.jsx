import { useState, useEffect } from 'react';
import { Heart, RefreshCw } from 'lucide-react';

// ImageWordle component with real API calls
const ImageWordle = () => {
  const [image, setImage] = useState(null);
  const [guess, setGuess] = useState("");
  const [message, setMessage] = useState("Welcome to ImageWordle! Start a new game to begin.");
  const [lives, setLives] = useState(5);
  const [difficulty, setDifficulty] = useState("Medium");
  const [gameOver, setGameOver] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [showFailure, setShowFailure] = useState(false);
  const [hint, setHint] = useState("Click 'Use Hint' to reveal a hint (costs 1 life)");
  const [guesses, setGuesses] = useState([]);
  const [feedback, setFeedback] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [targetWord, setTargetWord] = useState("");
  const [error, setError] = useState(null);
  
  // API base URL - should point to your Flask backend
  const API_URL = 'http://127.0.0.1:5000/api';
  
  // Generate a session ID on component mount
  useEffect(() => {
    setSessionId(`session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`);
  }, []);
  
  // Fallback mock hint functionality - renamed to avoid React hook naming confusion
  const applyMockHint = () => {
    if (lives <= 1) {
      setLives(0);
      setGameOver(true);
      setShowFailure(true);
      setTargetWord("CHAIR");
      setMessage("Game over! You're out of lives. The word was: CHAIR (MOCK DATA)");
    } else {
      setLives(lives - 1);
      const hints = [
        "This object is commonly found in households and is used for sitting.",
        "This item usually has four legs and a back for support.",
        "People use this when they want to sit at a table or desk."
      ];
      
      const hintsUsed = 5 - lives;
      if (hintsUsed < hints.length) {
        setHint(`Hint ${hintsUsed + 1}: ${hints[hintsUsed]} (MOCK DATA)`);
        setMessage(`Hint used! The image is now clearer. You have ${lives - 1} lives remaining. (MOCK DATA)`);
      } else {
        setHint("No more hints available. (MOCK DATA)");
        setMessage(`You've used all available hints. You have ${lives - 1} lives remaining. (MOCK DATA)`);
      }
    }
  };
  
  // Start a new game - USING REAL API
  const startNewGame = async () => {
    setLoading(true);
    setMessage("Starting a new game...");
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/newgame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          difficulty: difficulty
        }),
      });
      
      const data = await response.json();
      
      if (data.success) {
        // Convert base64 image to a displayable format
        const imageUrl = `data:image/png;base64,${data.image}`;
        
        setImage(imageUrl);
        setLives(data.lives);
        setGuesses([]);
        setFeedback([]);
        setGameOver(false);
        setShowSuccess(false);
        setShowFailure(false);
        setHint(data.hint);
        setMessage(data.message);
      } else {
        setError(data.message || "Failed to start a new game. Please try again.");
        // Fallback to mock data for testing
        setImage("/api/placeholder/512/512");
        setLives(5);
        setGuesses([]);
        setFeedback([]);
        setGameOver(false);
        setShowSuccess(false);
        setShowFailure(false);
        setHint("Click 'Use Hint' to reveal a hint (costs 1 life)");
        setMessage(`New game started! Difficulty: ${difficulty}. The secret word has 5-7 letters. You have 5 lives and 3 hints. (MOCK DATA)`);
      }
    } catch (error) {
      console.error("Error starting game:", error);
      setError("Network error. Please check your connection and try again.");
      
      // Fallback to mock data for testing
      setImage("/api/placeholder/512/512");
      setLives(5);
      setGuesses([]);
      setFeedback([]);
      setGameOver(false);
      setShowSuccess(false);
      setShowFailure(false);
      setHint("Click 'Use Hint' to reveal a hint (costs 1 life)");
      setMessage(`New game started! Difficulty: ${difficulty}. The secret word has 5-7 letters. You have 5 lives and 3 hints. (MOCK DATA)`);
    } finally {
      setLoading(false);
    }
  };
  
  // Submit a guess - USING REAL API
  const submitGuess = async () => {
    if (!guess.trim()) {
      setMessage("Please enter a word to guess.");
      return;
    }
    
    if (gameOver) {
      setMessage("Game is already over! Start a new game.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/guess`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId,
          guess: guess.toLowerCase().trim()
        }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (data.image) {
          const imageUrl = `data:image/png;base64,${data.image}`;
          setImage(imageUrl);
        }
        
        setGuesses(data.guesses);
        setFeedback(data.feedback);
        setLives(data.lives);
        setMessage(data.message);
        setHint(data.hint);
        
        if (data.game_over) {
          setGameOver(true);
          if (data.win) {
            setShowSuccess(true);
          } else {
            setTargetWord(data.target_word ? data.target_word.toUpperCase() : "UNKNOWN");
            setShowFailure(true);
          }
        }
      } else {
        setError(data.message || "Failed to process guess. Please try again.");
        // Continue with mock data as fallback
        processMockGuess(guess.toLowerCase());
      }
    } catch (error) {
      console.error("Error submitting guess:", error);
      setError("Network error. Please check your connection and try again.");
      // Continue with mock data as fallback
      processMockGuess(guess.toLowerCase());
    } finally {
      setGuess("");
      setLoading(false);
    }
  };
  
  // Use a hint - USING REAL API
  const useHint = async () => {
    if (gameOver) {
      setMessage("Game is already over! Start a new game.");
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/hint`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: sessionId
        }),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (data.image) {
          const imageUrl = `data:image/png;base64,${data.image}`;
          setImage(imageUrl);
        }
        
        setLives(data.lives);
        setMessage(data.message);
        setHint(data.hint);
        
        if (data.game_over) {
          setGameOver(true);
          if (data.win) {
            setShowSuccess(true);
          } else {
            setTargetWord(data.target_word ? data.target_word.toUpperCase() : "UNKNOWN");
            setShowFailure(true);
          }
        }
      } else {
        setError(data.message || "Failed to use hint. Please try again.");
        // Use mock hint as fallback
        applyMockHint();
      }
    } catch (error) {
      console.error("Error using hint:", error);
      setError("Network error. Please check your connection and try again.");
      // Use mock hint as fallback
      applyMockHint();
    } finally {
      setLoading(false);
    }
  };
  
  // Fallback mock functionality for testing when API is unavailable
  const processMockGuess = (guessLower) => {
    const correctWord = "chair"; // Example word
    
    // Check if the guess is correct
    if (guessLower === correctWord) {
      setGameOver(true);
      setShowSuccess(true);
      setMessage(`Congratulations! You guessed the word '${correctWord}' correctly!`);
      setGuesses([...guesses, guessLower]);
      setFeedback([...feedback, "🟩C🟩H🟩A🟩I🟩R"]);
    } else {
      // Process incorrect guess
      const newLives = lives - 1;
      setLives(newLives);
      
      // Generate mock feedback
      let feedbackStr = "";
      for (let i = 0; i < Math.min(guessLower.length, correctWord.length); i++) {
        if (guessLower[i] === correctWord[i]) {
          feedbackStr += `🟩${guessLower[i].toUpperCase()}`;
        } else if (correctWord.includes(guessLower[i])) {
          feedbackStr += `🟨${guessLower[i].toUpperCase()}`;
        } else {
          feedbackStr += `⬛${guessLower[i].toUpperCase()}`;
        }
      }
      
      setGuesses([...guesses, guessLower]);
      setFeedback([...feedback, feedbackStr]);
      
      if (newLives <= 0) {
        setGameOver(true);
        setShowFailure(true);
        setTargetWord(correctWord.toUpperCase());
        setMessage(`Game over! You're out of lives. The word was: ${correctWord.toUpperCase()}`);
      } else {
        setMessage(`Incorrect guess. Lives remaining: ${newLives} (MOCK DATA)`);
      }
    }
  };
  
  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && guess.trim()) {
      submitGuess();
    }
  };
  
  return (
    <div className="bg-gray-100 min-h-screen p-4">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md overflow-hidden">
        <div className="bg-blue-600 text-white p-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">ImageWordle</h1>
          <div className="flex items-center space-x-2">
            {Array(lives).fill().map((_, i) => (
              <Heart key={i} className="text-red-400 fill-red-400" size={24} />
            ))}
            {Array(5 - lives).fill().map((_, i) => (
              <Heart key={i + lives} className="text-red-200" size={24} />
            ))}
            <span className="ml-2 font-bold">{lives}</span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4">
          <div className="space-y-4">
            <div className="bg-gray-200 rounded-lg p-2 h-80 flex items-center justify-center">
              {image ? (
                <img 
                  src={image} 
                  alt="Pixelized word clue" 
                  className="max-h-full max-w-full object-contain"
                />
              ) : (
                <div className="text-gray-500">Start a new game to see an image</div>
              )}
            </div>
            
            <div className="flex gap-2">
              <select 
                className="bg-white border border-gray-300 rounded px-4 py-2 flex-grow"
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                disabled={loading}
              >
                <option value="Easy">Easy</option>
                <option value="Medium">Medium</option>
                <option value="Hard">Hard</option>
              </select>
              
              <button 
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded flex items-center gap-1"
                onClick={startNewGame}
                disabled={loading}
              >
                <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
                {loading ? "Starting..." : "New Game"}
              </button>
            </div>
            
            <div className="flex gap-2">
              <input
                type="text"
                value={guess}
                onChange={(e) => setGuess(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Enter your guess"
                className="border border-gray-300 rounded px-4 py-2 flex-grow"
                disabled={loading || gameOver}
              />
              
              <button 
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
                onClick={submitGuess}
                disabled={loading || gameOver || !guess.trim()}
              >
                {loading ? "Submitting..." : "Submit"}
              </button>
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg min-h-12">
              <div className="font-medium">{message}</div>
              {error && (
                <div className="mt-2 text-red-500 flex items-center gap-1">
                  <span>{error}</span>
                </div>
              )}
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded min-h-24">
              <h3 className="font-bold text-yellow-800">Hint</h3>
              <p className="text-yellow-700 whitespace-pre-wrap">{hint}</p>
            </div>
            
            <button 
              className="bg-yellow-500 hover:bg-yellow-600 text-white px-4 py-2 rounded w-full"
              onClick={useHint}
              disabled={loading || gameOver}
            >
              Use Hint (costs 1 life)
            </button>
            
            <div className="space-y-2">
              <h3 className="font-bold">Previous Guesses</h3>
              {guesses.length === 0 ? (
                <div className="text-gray-500">No guesses yet</div>
              ) : (
                <div className="space-y-2 max-h-60 overflow-y-auto bg-gray-50 p-2 rounded">
                  {guesses.map((guess, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <div className="uppercase font-mono">{guess}</div>
                      <div className="ml-4 font-mono text-lg">{feedback[index]}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            <div className="bg-gray-100 p-4 rounded-lg">
              <h3 className="font-bold mb-2">Game Help</h3>
              <ul className="list-disc pl-5 space-y-1 text-sm">
                <li>Guess the word from the pixelized image</li>
                <li>Each incorrect guess costs 1 life</li>
                <li>Use hints to make the image clearer (costs 1 life)</li>
                <li><span className="bg-green-200 px-1">🟩A</span> = correct letter in correct position</li>
                <li><span className="bg-yellow-200 px-1">🟨B</span> = correct letter in wrong position</li>
                <li><span className="bg-gray-200 px-1">⬛C</span> = letter not in the word</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      
      {/* Success Modal */}
      {showSuccess && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-md text-center relative">
            <button 
              className="absolute top-2 right-2 text-gray-500"
              onClick={() => setShowSuccess(false)}
            >
              ✕
            </button>
            <h2 className="text-3xl font-bold text-green-600 mb-4">Congratulations!</h2>
            <p className="text-xl mb-6">You successfully guessed the word!</p>
            <button 
              className="bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg text-lg"
              onClick={() => {
                setShowSuccess(false);
                startNewGame();
              }}
            >
              Play Again
            </button>
          </div>
        </div>
      )}
      
      {/* Failure Modal */}
      {showFailure && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-8 max-w-md text-center relative">
            <button 
              className="absolute top-2 right-2 text-gray-500"
              onClick={() => setShowFailure(false)}
            >
              ✕
            </button>
            <h2 className="text-3xl font-bold text-red-600 mb-4">Better Luck Next Time!</h2>
            <p className="text-xl mb-6">The word was <span className="font-bold">{targetWord}</span></p>
            <div className="flex gap-4 justify-center">
              <button 
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg text-lg"
                onClick={() => {
                  setShowFailure(false);
                  startNewGame();
                }}
              >
                Play Again
              </button>
              <button 
                className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-6 py-3 rounded-lg text-lg"
                onClick={() => {
                  setShowFailure(false);
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageWordle;
