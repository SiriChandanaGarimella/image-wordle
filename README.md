
# ImageWordle - AI Word Guessing Game

ImageWordle is a fun guessing game where you try to identify a word based on an AI-generated image. As you use hints, the blurry image becomes clearer!

## What the Game Does

- AI picks a random word and creates an image for it
- The image starts blurry or pixelated based on your chosen difficulty
- You make guesses to identify the word
- You can use hints (costs 1 life) to see the image more clearly
- Color-coded feedback shows which letters are correct

## AI Tools We Used

1. **For Word Generation**:
   - DistilGPT-2 (via Hugging Face) - Creates themed words based on selected categories (Alternative)

2. **For Images**: 
   - DALL-E 2 (via OpenAI API) 
   - Stable Diffusion (as backup)

3. **For Hints**: 
   - GPT-3.5 Turbo (via OpenAI API)

4. **For Understanding Guesses**: 
   - BART (for comparing your guesses to the answer)

## How to Run the Game

### What You Need
- Python 3.8 or newer
- Node.js and npm (for React frontend)
- OpenAI API key
- About 4GB of RAM

### Backend Setup

1. **Get the code:**
   ```bash
   git clone https://github.com/SiriChandanaGarimella/image-wordle
   cd image-wordle
   ```

2. **Create a Python virtual environment:**
   ```
   cd server
   ```

   **In Windows using Git Bash:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate
   ```

   **In macOS/Linux:**
   ```bash
   python -m venv venv or python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your API keys:**

   Create a file named `.env` in the main folder with:
   ```
   OPENAI_API_KEY="your_openai_key_here"
   HUGGINGFACE_TOKEN="your_huggingface_token_here"
   ```

   Or set them in the terminal:

   **In Windows using Git Bash:**
   ```bash
   export OPENAI_API_KEY="your_openai_key_here"
   export HUGGINGFACE_TOKEN="your_huggingface_token_here"
   ```

   **In macOS/Linux:**
   ```bash
   export OPENAI_API_KEY="your_openai_key_here"
   export HUGGINGFACE_TOKEN="your_huggingface_token_here"
   ```

5. **Start the Flask server:**
   ```bash
   python app.py
   or
   python3 app.py
   ```
   The backend will run on port 5000.

---

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install React dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm start
   ```
   This will start the React app on port 3000.

4. **Play the game:**

   Open your web browser and go to:
   ```
   http://localhost:3000
   ```

---

**If you need to modify the API endpoint in the frontend:**

Open `src/components/ImageWordle.jsx`  
Find this line:
```js
const API_URL = 'http://127.0.0.1:5000/api';
```
Change it if your backend is running on a different port or host.

---

### How to Play

1. Choose a difficulty level (**Easy**, **Medium**, **Hard**)
2. Click "New Game" to start
3. Type your word guesses in the input box
4. Use the "Use Hint" button when you need help (costs 1 life)
5. Watch for colored feedback on your guesses:

   - 🟩 **Green**: Right letter in right spot  
   - 🟨 **Yellow**: Right letter in wrong spot  
   - ⬛ **Black**: Letter not in the word

## Smart Caching System

The game implements an efficient caching system to improve performance and reduce API costs:

* **Image Caching**: Once an image is generated for a word, it's saved to the local disk. For future games with the same word, the cached image is reused instead of calling the API again.

* **Hint Caching**: All hints generated for words are stored in a JSON file (`clues.json`). This allows the game to instantly provide hints for previously seen words without additional API calls.

* **Automatic Fallbacks**: If API services are unavailable or rate-limited, the system automatically falls back to cached content, ensuring smooth gameplay even without an internet connection.
