import { useState, useEffect } from 'react';
import { Heart, RefreshCw, Info, X, CheckCircle, AlertCircle, Lightbulb, ChevronLeft, ChevronRight } from 'lucide-react';

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
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [wordLength, setWordLength] = useState(0);
  const [allHints, setAllHints] = useState([]);
  const [currentHintIndex, setCurrentHintIndex] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  
  // API base URL - should point to your Flask backend
  const API_URL = process.env.REACT_APP_API_URL || 'http://127.0.0.1:5000/api';
  
  // Generate a session ID on component mount
  useEffect(() => {
    setSessionId(`session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`);
  }, []);
  
  // Fallback mock hint functionality
  const applyMockHint = () => {
    if (lives <= 1) {
      setLives(0);
      setGameOver(true);
      setShowFailure(true);
      setTargetWord("CHAIR");
      setMessage("Game over! You're out of lives. The word was: CHAIR");
    } else {
      setLives(lives - 1);
      const hints = [
        "This object is commonly found in households and is used for sitting.",
        "This item usually has four legs and a back for support.",
        "People use this when they want to sit at a table or desk."
      ];
      
      const hintsUsed = 5 - lives;
      if (hintsUsed < hints.length) {
        setHint(`Hint ${hintsUsed + 1}: ${hints[hintsUsed]}`);
        setMessage(`Hint used! The image is now clearer. You have ${lives - 1} lives remaining.`);
      } else {
        setHint("No more hints available.");
        setMessage(`You've used all available hints. You have ${lives - 1} lives remaining.`);
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
        setWordLength(data.word_length);
        setAllHints([]);
        setGameStarted(true);
        setCurrentHintIndex(0);
      } else {
        setError(data.message || "Failed to start a new game. Please try again.");
        // Fallback to mock data for testing
        setImage("/api/placeholder/512/512");
        setLives(5);
        setGuesses([]);
        setFeedback([]);
        setShowSuccess(false);
        setShowFailure(false);
        setGameStarted(false);
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
      setGameStarted(false);
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
        setAllHints(prev => {
          const updated = [...prev, data.hint];
          setCurrentHintIndex(updated.length - 1);
          return updated;
        });
        
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
  
  // Mock guess processing
  const processMockGuess = (guessLower) => {
    const correctWord = "chair"; // Example word
    
    // Check if the guess is correct
    if (guessLower === correctWord) {
      setGameOver(true);
      setShowSuccess(true);
      setMessage(`Congratulations! You guessed the word '${correctWord.toUpperCase()}' correctly!`);
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
      
      // Add extra characters if the guess is longer than the correct word
      for (let i = correctWord.length; i < guessLower.length; i++) {
        feedbackStr += `⬛${guessLower[i].toUpperCase()}`;
      }
      
      setGuesses([...guesses, guessLower]);
      setFeedback([...feedback, feedbackStr]);
      
      if (newLives <= 0) {
        setGameOver(true);
        setShowFailure(true);
        setTargetWord(correctWord.toUpperCase());
        setMessage(`Game over! You're out of lives. The word was: ${correctWord.toUpperCase()}`);
      } else {
        setMessage(`Incorrect guess. Lives remaining: ${newLives}`);
      }
    }
  };
  
  return (
    <div className="bg-gradient-to-b from-black to-indigo-950 min-h-screen p-4 md:p-8 text-white font-sans">
      {/* Success Modal */}
      {showSuccess && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/70 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-emerald-600 to-teal-800 p-6 rounded-2xl max-w-md w-full shadow-xl border border-emerald-400/20 animate-fadeIn">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-2xl font-bold">Success!</h3>
              <button onClick={() => setShowSuccess(false)} className="text-white/80 hover:text-white">
                <X size={24} />
              </button>
            </div>
            <div className="flex items-center justify-center mb-6">
              <CheckCircle size={64} className="text-emerald-300" />
            </div>
            <p className="text-center text-xl mb-4">
              Congratulations! You've guessed the word correctly!
            </p>
            <button 
              onClick={() => {
                setShowSuccess(false);
                startNewGame();
              }}
              className="w-full bg-emerald-500 hover:bg-emerald-400 text-white py-3 rounded-xl font-bold transition-all"
            >
              Play Again
            </button>
          </div>
        </div>
      )}
      
      {/* Failure Modal */}
      {showFailure && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/70 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-rose-800 to-red-900 p-6 rounded-2xl max-w-md w-full shadow-xl border border-rose-500/20 animate-fadeIn">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-2xl font-bold">Game Over</h3>
              <button onClick={() => setShowFailure(false)} className="text-white/80 hover:text-white">
                <X size={24} />
              </button>
            </div>
            <div className="flex items-center justify-center mb-6">
              <AlertCircle size={64} className="text-rose-300" />
            </div>
            <p className="text-center text-xl mb-2">
              You've run out of lives!
            </p>
            <p className="text-center mb-4">
              The word was: <span className="font-mono font-bold text-2xl tracking-wider">{targetWord}</span>
            </p>
            <button 
              onClick={() => {
                setShowFailure(false);
                startNewGame();
              }}
              className="w-full bg-rose-600 hover:bg-rose-500 text-white py-3 rounded-xl font-bold transition-all"
            >
              Try Again
            </button>
          </div>
        </div>
      )}
      
      {/* Info Modal */}
      {showInfoModal && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/70 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-indigo-800 to-violet-900 p-6 rounded-2xl max-w-md w-full shadow-xl border border-indigo-500/20 animate-fadeIn">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-2xl font-bold">How to Play</h3>
              <button onClick={() => setShowInfoModal(false)} className="text-white/80 hover:text-white">
                <X size={24} />
              </button>
            </div>
            <div className="space-y-4">
              <p>
                <span className="font-semibold">Goal:</span> Guess the word represented by the blurry image before running out of lives.
              </p>
              <div>
                <span className="font-semibold">Difficulty levels:</span>
                <ul className="ml-6 mt-1 list-disc">
                  <li><span className="font-medium text-emerald-300">Easy:</span> Less blur, more common words</li>
                  <li><span className="font-medium text-yellow-300">Medium:</span> Moderate blur, standard words</li>
                  <li><span className="font-medium text-rose-300">Hard:</span> Heavy blur, challenging words</li>
                </ul>
              </div>
              <div>
                <span className="font-semibold">Feedback colors:</span>
                <div className="flex flex-wrap gap-2 mt-2">
                  <span className="bg-green-500 text-black px-2 py-1 rounded">🟩 Right letter, right position</span>
                  <span className="bg-yellow-500 text-black px-2 py-1 rounded">🟨 Right letter, wrong position</span>
                  <span className="bg-gray-500 text-white px-2 py-1 rounded">⬛ Wrong letter</span>
                </div>
              </div>
              <p>
                <span className="font-semibold">Hints:</span> Use hints to get clues about the word or make the image clearer, but each hint costs 1 life.
              </p>
            </div>
            <button 
              onClick={() => setShowInfoModal(false)}
              className="w-full bg-indigo-600 hover:bg-indigo-500 text-white py-3 rounded-xl font-bold transition-all mt-4"
            >
              Got it!
            </button>
          </div>
        </div>
      )}
      
      <div className="max-w-5xl mx-auto">
        {/* Header section */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-r from-indigo-500 to-purple-600 p-2 rounded-lg shadow-lg">
              <span className="text-3xl">🎨</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold bg-gradient-to-r from-indigo-400 via-purple-300 to-pink-400 bg-clip-text text-transparent">
              ImageWordle
            </h1>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="flex items-center">
              {Array(lives).fill().map((_, i) => (
                <Heart key={i} className="text-pink-500 fill-pink-500 -mr-1" size={20} />
              ))}
              {Array(5 - lives).fill().map((_, i) => (
                <Heart key={i + lives} className="text-gray-700 -mr-1" size={20} />
              ))}
              <span className="ml-3 font-semibold">{lives}</span>
            </div>
            
            <button 
              className="bg-indigo-900/50 hover:bg-indigo-800/70 p-2 rounded-full"
              onClick={() => setShowInfoModal(true)}
            >
              <Info size={20} />
            </button>
          </div>
        </div>
        
        <div className="grid md:grid-cols-5 gap-6">
          {/* Left column - Image and controls */}
          <div className="md:col-span-3 space-y-4">
            {/* Image container */}
            <div className="bg-gradient-to-br from-gray-900 to-indigo-900/50 rounded-2xl p-4 flex justify-center">
              <div className="rounded-xl overflow-hidden flex items-center justify-center border border-indigo-500/30 bg-black/30 shadow-inner">
                {image ? (
                  <img 
                    src={image} 
                    alt="Pixelized word clue" 
                    className="w-auto h-auto max-w-full max-h-96"
                  />
                ) : (
                  <div className="text-center p-8 w-64 h-64 flex flex-col items-center justify-center">
                    <div className="mb-4 text-indigo-300/70">
                      <RefreshCw size={48} className="mx-auto opacity-60" />
                    </div>
                    <p className="text-indigo-200">Start a new game to see the image!</p>
                  </div>
                )}
              </div>
            </div>
            
            {/* Controls */}
            <div className="flex flex-col sm:flex-row gap-3">
              <select 
                className="bg-indigo-950 border border-indigo-500/50 text-indigo-100 rounded-xl px-4 py-3 flex-grow shadow-inner focus:outline-none focus:ring-2 focus:ring-indigo-500/50"
                value={difficulty}
                onChange={(e) => setDifficulty(e.target.value)}
                disabled={loading}
              >
                <option value="Easy">Easy</option>
                <option value="Medium">Medium</option>
                <option value="Hard">Hard</option>
              </select>

              <button 
                className="bg-gradient-to-r from-green-600 to-emerald-700 hover:from-green-500 hover:to-emerald-600 text-white px-6 py-3 rounded-xl flex items-center justify-center gap-2 font-semibold shadow-lg transition-all"
                onClick={startNewGame}
                disabled={loading}
              >
                <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
                {loading ? "Starting..." : "New Game"}
              </button>
            </div>
            
            {/* Input and submit */}
            <div className="relative">
              <input
                type="text"
                value={guess}
                onChange={(e) => setGuess(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && submitGuess()}
                placeholder={wordLength ? `Enter a ${wordLength}-letter word` : "Enter your guess"}
                className="w-full px-5 py-4 pr-28 rounded-xl border border-indigo-500/30 bg-indigo-950/50 text-white shadow-inner focus:outline-none focus:ring-2 focus:ring-indigo-500/50 placeholder-indigo-300/50"
                disabled={loading || gameOver || !gameStarted}
                maxLength={15}
              />
              <button 
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-indigo-600 to-purple-700 hover:from-indigo-500 hover:to-purple-600 text-white px-4 py-2 rounded-lg shadow-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={submitGuess}
                disabled={loading || gameOver || !guess.trim() || !gameStarted}
              >
                Submit
              </button>
            </div>
            
            {/* Message area */}
            <div className={`p-4 rounded-xl shadow-inner ${error ? 'bg-red-900/30 border border-red-500/30' : 'bg-indigo-900/30 border border-indigo-500/20'}`}>
              <div className="font-medium">{message}</div>
              {error && <div className="text-red-300 mt-2 text-sm">{error}</div>}
            </div>
          </div>
          
          {/* Right column - Hints and guesses */}
          <div className="md:col-span-2 space-y-4">
            {/* Hint box */}
            <div className="bg-gradient-to-br from-indigo-900/70 to-purple-900/70 rounded-xl p-4 border border-indigo-500/30 shadow-lg">
              <div className="flex items-center gap-2 mb-2">
                <Lightbulb size={18} className="text-yellow-400" />
                <h3 className="font-bold text-lg text-yellow-100">Hint</h3>
                {allHints.length > 0 && (
                  <span className="text-xs text-indigo-300 ml-auto">
                    {currentHintIndex + 1} / {allHints.length}
                  </span>
                )}
              </div>

              <p className="whitespace-pre-wrap text-sm mt-2 text-indigo-100">
                {allHints.length > 0 ? allHints[currentHintIndex] : hint}
              </p>

              {allHints.length > 1 && (
                <div className="flex items-center justify-center gap-4 mt-3">
                  <button
                    onClick={() => setCurrentHintIndex(prev => Math.max(0, prev - 1))}
                    disabled={currentHintIndex === 0}
                    className="p-1.5 rounded-lg bg-indigo-800/50 hover:bg-indigo-700/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    <ChevronLeft size={18} />
                  </button>
                  <div className="flex gap-1.5">
                    {allHints.map((_, i) => (
                      <button
                        key={i}
                        onClick={() => setCurrentHintIndex(i)}
                        className={`w-2 h-2 rounded-full transition-all ${
                          i === currentHintIndex ? 'bg-yellow-400' : 'bg-indigo-600'
                        }`}
                      />
                    ))}
                  </div>
                  <button
                    onClick={() => setCurrentHintIndex(prev => Math.min(allHints.length - 1, prev + 1))}
                    disabled={currentHintIndex === allHints.length - 1}
                    className="p-1.5 rounded-lg bg-indigo-800/50 hover:bg-indigo-700/50 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                  >
                    <ChevronRight size={18} />
                  </button>
                </div>
              )}

              <button
                className="mt-4 bg-gradient-to-r from-yellow-600 to-amber-700 hover:from-yellow-500 hover:to-amber-600 text-white w-full px-4 py-3 rounded-xl font-semibold shadow-lg flex items-center justify-center gap-2 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                onClick={useHint}
                disabled={loading || gameOver || !gameStarted}
              >
                <Lightbulb size={18} />
                Use Hint ({2 - allHints.length} remaining)
              </button>
            </div>
            
            {/* Guesses */}
            <div className="bg-gradient-to-br from-gray-900 to-indigo-900/50 rounded-xl p-4 border border-indigo-500/30 shadow-lg">
            <div className="flex justify-between items-center mb-3">
              <h3 className="font-bold text-lg">Your Guesses</h3>
              {wordLength > 0 && (
                <span className="bg-indigo-600/50 text-indigo-200 text-sm px-3 py-1 rounded-full border border-indigo-500/30">
                  {wordLength} letters
                </span>
              )}
            </div>
              {guesses.length === 0 ? (
                <p className="text-indigo-300/70 text-center py-4">No guesses yet</p>
              ) : (
                <ul className="space-y-2 max-h-80 overflow-y-auto pr-2">
                  {guesses.map((guess, i) => (
                    <li 
                      key={i} 
                      className="flex justify-between items-center bg-indigo-900/30 p-3 rounded-lg border border-indigo-600/20"
                    >
                      <span className="font-mono uppercase tracking-wide">{guess}</span>
                      <span className="font-mono text-lg tracking-wider">{feedback[i]}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageWordle;