from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import base64
from io import BytesIO
import time
import os
import hashlib
import json
import re
import requests
from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import pipeline
import openai

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration constants
CLUE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "clues.json")
# Categories for word generation
CATEGORIES = [
    "Animals", "Food", "Nature", "Objects", "Places", 
    "Sports", "Technology", "Transportation", "Weather"
]

# List of words with 5-7 letters
WORD_LIST = [
    # 5 letter words
    "beach", "clock", "plant", "space", "chair", "table", "mouse", "house", "fruit",
    "tiger", "robot", "music", "dance", "train", "cloud", "river", "horse", "smile",
    # 6 letter words
    "garden", "castle", "coffee", "bridge", "dragon", "planet", "sunset", "market",
    "winter", "camera", "laptop", "forest", "island", "guitar", "soccer", "turtle",
    # 7 letter words
    "diamond", "mountain", "rainbow", "bicycle", "library", "stadium", "dolphin",
    "pyramid", "penguin", "volcano", "elephant", "tornado", "sunflower", "compass"
]

# Word categories for better hint generation
WORD_CATEGORIES = {
    "beach": "nature", "clock": "object", "plant": "nature", "space": "concept",
    "chair": "furniture", "table": "furniture", "mouse": "animal", "house": "building",
    "fruit": "food", "tiger": "animal", "robot": "technology", "music": "art",
    "dance": "activity", "train": "transportation", "cloud": "nature", "river": "nature",
    "horse": "animal", "smile": "expression", "garden": "nature", "castle": "building",
    "coffee": "food", "bridge": "structure", "dragon": "mythical", "planet": "astronomy",
    "sunset": "nature", "market": "place", "winter": "season", "camera": "technology",
    "laptop": "technology", "forest": "nature", "island": "geography", "guitar": "music",
    "soccer": "sport", "turtle": "animal", "diamond": "mineral", "mountain": "geography",
    "rainbow": "nature", "bicycle": "transportation", "library": "building",
    "stadium": "building", "dolphin": "animal", "pyramid": "structure", "penguin": "animal",
    "volcano": "geography", "elephant": "animal", "tornado": "weather",
    "sunflower": "plant", "compass": "tool"
}

# Game state storage
game_states = {}

# Global model instances
image_model = None
text_classifier = None
text_generator = None

def generate_random_word(category):
    generator = pipeline('text-generation', 
                         model='distilgpt2', 
                         token=os.environ.get("HUGGINGFACE_TOKEN"))
    prompt = f"Give me a single 5 to 7 letter word based on the category \"{category}\": "
    responses = generator(prompt, 
                     max_length=len(prompt.split()) + 5,
                     num_return_sequences=5,
                     temperature=0.9,
                     truncation=True)
    for response in responses:
        generated_text = response['generated_text'][len(prompt):].strip()
        words = generated_text.split()
        
        for word in words:
            clean_word = ''.join(c for c in word if c.isalpha())
            
            if clean_word and 5<= len(clean_word) <= 7:
                return clean_word.lower()
    return random.choice(WORD_LIST)

def get_cache_directory():
    """Create and return the cache directory path"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def get_cached_image_path(word):
    """Generate a unique filename for a cached image"""
    word_hash = hashlib.md5(word.encode()).hexdigest()
    cache_dir = get_cache_directory()
    return os.path.join(cache_dir, f"{word_hash}.png")

def save_image_to_cache(image, word):
    """Save an image to the cache"""
    try:
        cache_path = get_cached_image_path(word)
        image.save(cache_path, format="PNG")
        print(f"Image for '{word}' saved to cache")
        return True
    except Exception as e:
        print(f"Error saving image to cache: {e}")
        return False

def load_image_from_cache(word):
    """Try to load an image from the cache"""
    try:
        cache_path = get_cached_image_path(word)
        if os.path.exists(cache_path):
            image = Image.open(cache_path)
            print(f"Image for '{word}' loaded from cache")
            return image
        return None
    except Exception as e:
        print(f"Error loading image from cache: {e}")
        return None

def get_random_cached_image():
    """Get a random image from the cache"""
    try:
        cache_dir = get_cache_directory()
        image_files = [f for f in os.listdir(cache_dir) if f.endswith('.png')]
        
        if not image_files:
            print("No cached images available")
            return None
        
        random_image_file = random.choice(image_files)
        image_path = os.path.join(cache_dir, random_image_file)
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading random cached image: {e}")
        return None

def initialize_models():
    """Initialize image generation and text models"""
    global image_model, text_classifier, text_generator

    try:
        print("Loading image generation model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Ensure no CUDA usage

        image_model = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )

        image_model.to("cpu")
        image_model.enable_attention_slicing(1)
        
        try:
            image_model.scheduler = DDIMScheduler.from_config(image_model.scheduler.config)
        except Exception as e:
            print(f"Could not set DDIM scheduler: {e}")
    except Exception as e:
        print(f"Error loading image generation model: {e}")
        image_model = None

    try:
        print("Loading BART zero-shot classification model...")
        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
        
        # Use smaller version if memory is a concern
        model_name = "facebook/bart-large-mnli"  # Can also use "facebook/bart-base-mnli" if memory is limited
        
        # Load tokenizer and model with proper settings
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Configure for CPU or GPU
        device = 0 if torch.cuda.is_available() else -1
        
        # Create the classification pipeline
        text_classifier = pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        print("BART classification model loaded successfully!")
    except Exception as e:
        print(f"Error loading BART classification model: {e}")
        # Define fallback only if BART fails
        def simple_classifier(text, labels, hypothesis_template=None):
            print(f"Using fallback classifier for: {text}, {labels}")
            scores = [sum(1 for c in text.lower() if c in label.lower()) / max(len(text), len(label)) for label in labels]
            return {"scores": scores, "labels": labels, "sequence": text}
        text_classifier = lambda text, labels, hypothesis_template=None: simple_classifier(text, labels)
        print("Fallback text classifier initialized")

def generate_image_with_model(word):
    """Generate an image using the local Stable Diffusion model"""
    try:
        if image_model is None:
            raise ValueError("Image model not available")

        # Generate prompt based on category
        category = WORD_CATEGORIES.get(word, "object")
        if category == "animal":
            prompt = f"A clear, detailed photograph of a {word}, wildlife photography, centered, realistic"
        elif category == "food":
            prompt = f"An appetizing photograph of {word}, food photography, detailed, centered"
        elif category == "nature":
            prompt = f"A beautiful landscape photograph showing {word}, nature photography, high quality"
        else:
            prompt = f"A clear, detailed photograph of a {word}, centered, high quality, realistic"

        # CPU-friendly settings
        num_steps = 10
        guidance_scale = 5.0
        height, width = 384, 384

        # Run generation
        with torch.no_grad():
            result = image_model(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                output_type="pil"
            )

        image = result.images[0]

        # Add simple watermark/hash
        word_hash = sum(ord(c) for c in word)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), f"ID: {word_hash % 1000}", fill=(255, 255, 255))

        return image
    except Exception as e:
        print(f"Error generating image with local model: {e}")
        return None

def generate_image_with_dalle2(word, api_key=None, max_retries=2):
    """Generate an image using OpenAI's DALL-E 2 API"""
    # Check cache first
    cached_image = load_image_from_cache(word)
    if cached_image:
        return cached_image
    
    # Get API key
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        random_image = get_random_cached_image()
        return random_image if random_image else generate_fallback_image(word)
    
    retries = 0
    while retries <= max_retries:
        try:
            # Create a prompt based on the word and category
            category = WORD_CATEGORIES.get(word, "object")
            
            if category == "animal":
                prompt = f"A clear, detailed photograph of a {word}, wildlife photography, centered, realistic"
            elif category == "food":
                prompt = f"An appetizing photograph of {word}, food photography, detailed, centered"
            elif category == "nature":
                prompt = f"A beautiful landscape photograph showing {word}, nature photography, high quality"
            else:
                prompt = f"A clear, detailed photograph of a {word}, centered, high quality, realistic"
            
            # Set up API request
            url = "https://api.openai.com/v1/images/generations"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": "dall-e-2",
                "prompt": prompt,
                "n": 1,
                "size": "256x256",
                "response_format": "url"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                response_data = response.json()
                image_url = response_data["data"][0]["url"]
                
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    image = Image.open(BytesIO(image_response.content))
                    
                    # Add a watermark
                    word_hash = sum(ord(c) for c in word)
                    draw = ImageDraw.Draw(image)
                    draw.text((10, 10), f"ID: {word_hash % 1000}", fill=(255, 255, 255))
                    
                    save_image_to_cache(image, word)
                    return image
            
            # Handle rate limits
            if response.status_code == 429:
                time.sleep(2 ** retries + 1)  # Exponential backoff
                
            retries += 1
            
        except Exception as e:
            print(f"Error with DALL-E 2: {e}")
            retries += 1
            if retries <= max_retries:
                time.sleep(2 ** retries)
    
    # Fallback if all attempts failed
    random_image = get_random_cached_image()
    return random_image if random_image else generate_fallback_image(word)

def generate_fallback_image(word):
    """Generate a simple geometric image if all methods fail"""
    try:
        # Create a blank image with text
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img.fill(255)  # White background
        
        # Add visual elements based on the word
        word_hash = sum(ord(c) for c in word)
        color1 = (word_hash % 200, (word_hash * 2) % 200, (word_hash * 3) % 200)
        color2 = ((word_hash * 4) % 200, (word_hash * 5) % 200, (word_hash * 6) % 200)
        
        # Draw shapes
        cv2.rectangle(img, (50, 50), (450, 450), color1, -1)
        cv2.rectangle(img, (100, 100), (400, 400), color2, -1)
        
        # Add hash ID, not the word itself
        cv2.putText(img, f"Word ID: {word_hash % 1000}", (150, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return Image.fromarray(img)
    except Exception as e:
        print(f"Error generating fallback image: {e}")
        # Even simpler fallback
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img.fill(200)  # Gray background
        cv2.putText(img, "Image Error", (150, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return Image.fromarray(img)

def generate_image(word):
    """Generate an image based on the word with caching and fallbacks"""
    try:
        # Check cache first
        cached_image = load_image_from_cache(word)
        if cached_image:
            return cached_image
        
        # Try DALL-E 2 API first if key is available
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            dalle2_image = generate_image_with_dalle2(word, openai_key)
            if dalle2_image:
                return dalle2_image
        
        # Try local model if available
        if image_model is not None:
            local_image = generate_image_with_model(word)
            if local_image is not None:
                save_image_to_cache(local_image, word)
                return local_image
        
        # Try random cached image as fallback
        random_image = get_random_cached_image()
        if random_image:
            return random_image
            
        # Last resort fallback
        fallback = generate_fallback_image(word)
        save_image_to_cache(fallback, word) 
        return fallback
        
    except Exception as e:
        print(f"Error in generate_image: {e}")
        return generate_fallback_image(word)

def apply_obscuring(image, difficulty_level, obscure_type="pixelize"):
    """Apply obscuring effect to image based on difficulty level"""
    try:
        if obscure_type == "blur":
            # Set blur parameters based on difficulty
            blur_amount = 21 if difficulty_level == "Easy" else 35 if difficulty_level == "Medium" else 51
            
            open_cv_image = np.array(image)
            if len(open_cv_image.shape) == 2:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)
            
            blur_amount = max(1, blur_amount)
            if blur_amount % 2 == 0:
                blur_amount += 1
            
            blurred = cv2.GaussianBlur(open_cv_image, (blur_amount, blur_amount), 0)
            return Image.fromarray(blurred)
        else:  # pixelize
            # Set block size based on difficulty
            block_size = 15 if difficulty_level == "Easy" else 25 if difficulty_level == "Medium" else 40
            
            open_cv_image = np.array(image)
            if len(open_cv_image.shape) == 2:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2RGB)
            
            height, width = open_cv_image.shape[:2]
            small_height = height // block_size
            small_width = width // block_size
            
            temp = cv2.resize(open_cv_image, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
            
            return Image.fromarray(pixelated)
    except Exception as e:
        print(f"Error applying obscuring effect: {e}")
        return image

def load_clue_cache():
    """Load previously cached clues from disk"""
    if not os.path.exists(CLUE_CACHE_PATH):
        return {}
    try:
        with open(CLUE_CACHE_PATH, "r") as f:
            return json.load(f)
    except:
        return {}

def save_clue_cache(clue_cache):
    """Save clue cache to disk"""
    with open(CLUE_CACHE_PATH, "w") as f:
        json.dump(clue_cache, f, indent=2)

clue_cache = load_clue_cache()

def generate_dynamic_hints(word, word_category):
    """Generate hints for a word when API-based hints are unavailable"""
    try:
        category = word_category.lower()
        
        # Word-specific templates for common words
        word_templates = {
            "chair": [
                "You sit on this object.",
                "It typically has four legs and a back for support.",
                "Found in dining rooms, classrooms, and offices."
            ],
            "beach": [
                "A sandy area where water meets land.",
                "People often visit here for swimming and sunbathing.",
                "It's a coastal feature popular for vacations."
            ],
            # ... more word templates would be here
        }
        
        # Category-based templates for general hints
        category_templates = {
            "nature": [
                "This is found in the natural world.",
                "It's a part of our environment.",
                "You might encounter this in the outdoors."
            ],
            "object": [
                "This is a physical item you can touch.",
                "It's something that can be used by people.",
                "You might find this in a home or office."
            ],
            # ... more category templates would be here
        }
        
        # Use word-specific hints if available
        if word in word_templates:
            specific_hints = word_templates[word]
        # Otherwise use category-based hints
        elif category in category_templates:
            specific_hints = category_templates[category]
        else:
            # Generate generic hints
            adjectives = ["interesting", "common", "notable", "distinctive", "recognizable"]
            contexts = ["everyday life", "many settings", "various situations", "different places", "our world"]
            actions = ["encounter", "recognize", "use", "see", "interact with"]
            
            specific_hints = [
                f"This {len(word)}-letter word refers to something {random.choice(adjectives)}.",
                f"You've likely {random.choice(actions)} this in {random.choice(contexts)}.",
                f"Think about objects or concepts that have {len(word)} letters."
            ]
        
        # Always add the length hint
        length_hints = [
            f"The word has exactly {len(word)} letters.",
            f"You're looking for a {len(word)}-letter word.",
            f"This word contains {len(word)} letters."
        ]
        specific_hints.append(random.choice(length_hints))
        
        # Ensure we don't have duplicate hints
        return list(dict.fromkeys(specific_hints))
    
    except Exception as e:
        print(f"Error generating dynamic hints: {e}")
        # Fallback hints
        return [
            f"This {len(word)}-letter word is something you might recognize.",
            f"It's a common thing that many people are familiar with.",
            f"The word has exactly {len(word)} letters."
        ]

def generate_clues(word, num_clues=3, difficulty="Medium"):
    """Generate clues for the given word based on difficulty level"""
    # Check if already cached
    if word in clue_cache and difficulty in clue_cache[word]:
        return clue_cache[word][difficulty]

    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Define clue styles based on difficulty level
        if difficulty == "Easy":
            system_prompt = "You are a helpful assistant that generates clear, obvious clues for a word guessing game. The clues should be easy to understand. Never use the target word in the clue."
            clue_styles = [
                f"Describe what this object or concept is in a very clear, simple sentence. Do not include the word '{word}'.",
                f"Give a very obvious use or function of this object in a simple sentence. Avoid using the word '{word}'.",
                f"Explain a key feature that makes this object easily identifiable. Do not mention '{word}'."
            ]
            temperature = 0.8
        elif difficulty == "Medium":
            system_prompt = "You are a helpful assistant that generates moderate clues for a word guessing game. The clues should be somewhat challenging but still clear. Never use the target word in the clue."
            clue_styles = [
                f"Describe what this object or concept is in a helpful, single sentence. Do not include the word '{word}'.",
                f"Give one full sentence explaining how this object is commonly used. Avoid using the word '{word}'.",
                f"Give a distinctive feature of this object in one complete sentence. Do not mention '{word}'."
            ]
            temperature = 0.9
        else:  # Hard
            system_prompt = "You are a helpful assistant that generates cryptic, challenging clues for a word guessing game. The clues should be difficult to decipher. Never use the target word in the clue."
            clue_styles = [
                f"Provide a subtle, somewhat abstract description of this concept without revealing too much. Do not include the word '{word}'.",
                f"Describe an uncommon or secondary use of this object in one sentence. Do not use the word '{word}'.",
                f"Mention an obscure detail or property of this object. Avoid directly naming '{word}'."
            ]
            temperature = 1.1

        clues = []
        for i in range(num_clues):
            user_prompt = clue_styles[i % len(clue_styles)]
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a {difficulty.lower()}-level clue for the word '{word}'. {user_prompt}"}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=60,
                temperature=temperature
            )

            reply = response["choices"][0]["message"]["content"].strip()
            clue = re.split(r'[.!?]', reply)[0].strip()

            if not clue.endswith(('.', '!', '?')):
                clue += '.'
            if word.lower() in clue.lower():
                clue = "Think of an object that's commonly used."

            clues.append(clue)

        # Cache the clues by difficulty
        if word not in clue_cache:
            clue_cache[word] = {}
        if difficulty == "Hard":
            clue_cache[word][difficulty] = clues
            save_clue_cache(clue_cache)
        return clues

    except Exception as e:
        print(f"Falling back to dynamic hint generation due to error: {e}")
        word_category = WORD_CATEGORIES.get(word, "object")
        return generate_dynamic_hints(word, word_category)

def generate_word_feedback(guess, target):
    """Generate Wordle-style feedback with character display"""
    try:
        if len(guess) != len(target):
            return f"Word must be exactly {len(target)} letters!"
        
        feedback = ["⬛"] * len(target)  # Default gray
        used_indices = set()
        
        # First pass: Mark correct positions (green)
        for i in range(len(target)):
            if guess[i] == target[i]:
                feedback[i] = f"🟩{guess[i].upper()}"  # Green with character
                used_indices.add(i)
        
        # Second pass: Mark correct letters in wrong positions (yellow)
        for i in range(len(guess)):
            if feedback[i] == "⬛":  # Only consider unmarked positions
                # Check if this letter appears in target at an unused position
                for j in range(len(target)):
                    if j not in used_indices and guess[i] == target[j]:
                        feedback[i] = f"🟨{guess[i].upper()}"  # Yellow with character
                        used_indices.add(j)
                        break
                
                # If still unmarked, add character to black square
                if feedback[i] == "⬛":
                    feedback[i] = f"⬛{guess[i].upper()}"
        
        return ''.join(feedback)
    except Exception as e:
        print(f"Error generating word feedback: {e}")
        # Return all gray with characters if error
        return ''.join([f"⬛{c.upper()}" for c in guess])

def image_to_base64(image):
    """Convert PIL image to base64 string for transmission"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def create_new_game(session_id, difficulty):
    """Create a new game state for the given session"""
    try:
        # Select a random word
        target_word = random.choice(WORD_LIST)
        # target_word = generate_random_word(random.choice(CATEGORIES)) # uncomment to use categories and generate words using DistilGPT2
        
        # Get the word's category for better hint generation
        word_category = WORD_CATEGORIES.get(target_word, "object")
        
        # Generate an image for the word
        full_image = generate_image(target_word)
        
        # Apply obscuring effect based on difficulty
        obscure_type = "pixelize"
        blurred_image = apply_obscuring(full_image, difficulty, obscure_type)
        
        # Generate hints specific to the word and difficulty
        hints = generate_clues(target_word, num_clues=3, difficulty=difficulty)
        
        # Initialize game state
        game_states[session_id] = {
            "target_word": target_word,
            "word_category": word_category,
            "difficulty": difficulty,
            "lives": 5,
            "max_guesses": 5,
            "guesses": [],
            "feedback": [],
            "hint_index": 0,
            "available_hints": 3,
            "hints": hints,
            "full_image": full_image,
            "current_image": blurred_image,
            "obscure_type": obscure_type,
            "game_over": False,
            "success": False,
            "created_at": time.time()
        }
        
        # Return initial game state
        return {
            "success": True,
            "message": f"New game started! Difficulty: {difficulty}. The secret word has {len(target_word)} letters. You have 5 lives and 3 hints.",
            "image": image_to_base64(blurred_image),
            "word_length": len(target_word),
            "guesses": [],
            "feedback": [],
            "lives": 5,
            "hint": "Click 'Use Hint' to reveal a hint (costs 1 life)",
            "game_over": False
        }
    except Exception as e:
        print(f"Error creating new game: {e}")
        return {
            "success": False,
            "message": f"Error creating new game: {str(e)}",
            "image": None,
            "lives": 5,
            "hint": "Error starting game",
            "game_over": False
        }

def process_guess(session_id, guess_word):
    """Process a guess for the current game using BART for similarity evaluation"""
    try:
        # Check if session exists
        if session_id not in game_states:
            return {
                "success": False,
                "message": "No active game. Please start a new game.",
                "lives": 0,
                "game_over": True
            }
        
        # Get game state
        game = game_states[session_id]
        
        # Check if game is already over
        if game["game_over"]:
            return {
                "success": False,
                "message": "Game is already over. Please start a new game.",
                "image": image_to_base64(game["full_image"]),
                "guesses": game["guesses"],
                "feedback": game["feedback"],
                "lives": game["lives"],
                "hint": "Game Over",
                "game_over": True
            }
        
        # Clean the guess
        guess = guess_word.lower().strip()
        
        # Check if word is correct length
        if len(guess) != len(game["target_word"]):
            return {
                "success": False,
                "message": f"Your guess must be {len(game['target_word'])} letters long!",
                "image": image_to_base64(game["current_image"]),
                "guesses": game["guesses"],
                "feedback": game["feedback"],
                "lives": game["lives"],
                "hint": "Click 'Use Hint' to reveal a hint (costs 1 life)" if game["hint_index"] == 0 else f"Hint: {game['hints'][game['hint_index']-1]}",
                "game_over": False
            }
        
        # Check if the guess is correct
        if guess == game["target_word"]:
            game["game_over"] = True
            game["success"] = True
            
            return {
                "success": True,
                "message": f"Congratulations! You guessed the word '{game['target_word']}' correctly!",
                "image": image_to_base64(game["full_image"]),
                "guesses": game["guesses"] + [guess],
                "feedback": game["feedback"] + [generate_word_feedback(guess, game["target_word"])],
                "lives": game["lives"],
                "hint": "You won!",
                "game_over": True,
                "win": True,
                "target_word": game["target_word"]
            }
        
        # Process incorrect guess
        game["guesses"].append(guess)
        feedback = generate_word_feedback(guess, game["target_word"])
        game["feedback"].append(feedback)
        
        # Reduce lives
        game["lives"] -= 1
        
        # Check if out of lives
        if game["lives"] <= 0:
            game["game_over"] = True
            
            return {
                "success": False,
                "message": f"Game over! You're out of lives. The word was: {game['target_word'].upper()}",
                "image": image_to_base64(game["full_image"]),
                "guesses": game["guesses"],
                "feedback": game["feedback"],
                "lives": 0,
                "hint": "Game Over",
                "game_over": True,
                "win": False,
                "target_word": game["target_word"]
            }
        
        # Use BART to evaluate similarity and generate appropriate feedback
        try:
            # Create context for better classification
            context = f"The word '{guess}' compared to the word '{game['target_word']}'"
            
            # Define possible feedback categories with nuanced descriptions
            feedback_categories = [
                "very similar and close to correct", 
                "somewhat similar with some matching elements", 
                "slightly similar but mostly different",
                "not similar at all"
            ]
            
            # Use BART to classify with the enhanced context
            result = text_classifier(
                context, 
                feedback_categories,
                hypothesis_template="These words are {}."
            )
            
            # Get the highest scoring category
            highest_category = result["labels"][0]
            confidence = result["scores"][0]
            
            # More varied hints based on BART classification
            if "very similar" in highest_category:
                hint_options = [
                    "Very close!", 
                    "You're almost there!", 
                    "That's really close to the answer!"
                ]
                hint_text = random.choice(hint_options)
            elif "somewhat similar" in highest_category:
                hint_options = [
                    "You're on the right track.", 
                    "Getting warmer.", 
                    "That's in the right direction."
                ]
                hint_text = random.choice(hint_options)
            elif "slightly similar" in highest_category:
                hint_options = [
                    "Somewhat related.", 
                    "A bit similar, but not quite.", 
                    "There's a slight connection."
                ]
                hint_text = random.choice(hint_options)
            else:
                hint_options = [
                    "Not really close.", 
                    "Try something different.", 
                    "That's quite far from the answer."
                ]
                hint_text = random.choice(hint_options)
            
            # Add confidence-based variation for more natural responses
            if confidence > 0.85:
                confidence_options = [
                    " I'm certain of it.",
                    " Definitely.",
                    " No doubt about it."
                ]
                hint_text += random.choice(confidence_options)
            elif confidence < 0.6:
                confidence_options = [
                    "Perhaps ",
                    "I think ",
                    "It seems "
                ]
                hint_text = random.choice(confidence_options) + hint_text.lower()
                
        except Exception as e:
            print(f"Error using BART for similarity: {e}")
            # Fallback to simple character overlap method
            guess_chars = set(guess)
            target_chars = set(game["target_word"])
            overlap = len(guess_chars.intersection(target_chars))
            similarity = overlap / max(len(guess_chars), len(target_chars))
            
            # Generate hint based on similarity
            if similarity > 0.8:
                hint_text = "Very close!"
            elif similarity > 0.6:
                hint_text = "You're on the right track."
            elif similarity > 0.4:
                hint_text = "Somewhat related."
            else:
                hint_text = "Not really close."
        
        message = f"Incorrect guess: {hint_text} Lives remaining: {game['lives']}"
        
        # Show current hint if available
        hint_display = "Click 'Use Hint' to reveal a hint (costs 1 life)" if game["hint_index"] == 0 else f"Hint: {game['hints'][game['hint_index']-1]}"
        
        return {
            "success": False,
            "message": message,
            "image": image_to_base64(game["current_image"]),
            "guesses": game["guesses"],
            "feedback": game["feedback"],
            "lives": game["lives"],
            "hint": hint_display,
            "game_over": False
        }
    except Exception as e:
        print(f"Error processing guess: {e}")
        return {
            "success": False,
            "message": f"Error processing guess: {str(e)}",
            "game_over": False
        }

def use_hint(session_id):
    """Use a hint to make image clearer and provide word hint"""
    try:
        # Check if session exists
        if session_id not in game_states:
            return {
                "success": False,
                "message": "No active game. Please start a new game.",
                "lives": 0,
                "hint": "No active game",
                "game_over": True
            }
        
        # Get game state
        game = game_states[session_id]
        
        # Check if game is already over
        if game["game_over"]:
            return {
                "success": False,
                "message": "Game is already over. Please start a new game.",
                "image": image_to_base64(game["full_image"]),
                "lives": game["lives"],
                "hint": "Game Over",
                "game_over": True
            }
        
        # Check if hints are available
        if game["available_hints"] <= 0:
            return {
                "success": False,
                "message": "You've used all your hints!",
                "image": image_to_base64(game["current_image"]),
                "lives": game["lives"],
                "hint": "No hints remaining. You've seen all available hints.",
                "game_over": False
            }
        
        # Use a hint and reduce lives
        game["available_hints"] -= 1
        game["lives"] -= 1
        
        # Check if out of lives
        if game["lives"] <= 0:
            game["game_over"] = True
            
            return {
                "success": False,
                "message": f"Game over! You're out of lives. The word was: {game['target_word'].upper()}",
                "image": image_to_base64(game["full_image"]),
                "lives": 0,
                "hint": "Game Over",
                "game_over": True,
                "win": False,
                "target_word": game["target_word"]
            }
        
        # Increment hint index
        game["hint_index"] += 1
        
        # Adjust difficulty based on hints used
        current_difficulty = "Hard"
        if game["hint_index"] == 1:
            current_difficulty = "Medium"  # First hint moves to medium difficulty
        elif game["hint_index"] >= 2:
            current_difficulty = "Easy"    # Second hint moves to easy difficulty
        
        # Apply new obscuring effect
        game["current_image"] = apply_obscuring(game["full_image"], current_difficulty, game["obscure_type"])
        
        # Get the current hint
        hint_text = game["hints"][game["hint_index"]-1] if game["hint_index"] <= len(game["hints"]) else "No more hints available."
        hint_message = f"Hint {game['hint_index']}: {hint_text}"
        
        return {
            "success": True,
            "message": f"Hint used! The image is now clearer. You have {game['lives']} lives remaining.",
            "image": image_to_base64(game["current_image"]),
            "lives": game["lives"],
            "hint": hint_message,
            "game_over": False
        }
    except Exception as e:
        print(f"Error using hint: {e}")
        return {
            "success": False,
            "message": f"Error using hint: {str(e)}",
            "game_over": False
        }

def cleanup_old_games():
    """Remove game states that are more than 1 hour old"""
    current_time = time.time()
    to_remove = []
    
    for session_id, game in game_states.items():
        if current_time - game.get("created_at", 0) > 3600:  # 1 hour
            to_remove.append(session_id)
    
    for session_id in to_remove:
        del game_states[session_id]
    
    print(f"Cleaned up {len(to_remove)} old game states")

# API Routes
@app.route('/api/newgame', methods=['POST'])
def new_game():
    """Start a new game"""
    try:
        data = request.json
        session_id = data.get('session_id', str(time.time()))
        difficulty = data.get('difficulty', 'Medium')
        
        result = create_new_game(session_id, difficulty)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error creating new game: {str(e)}"
        })

@app.route('/api/guess', methods=['POST'])
def guess():
    """Process a guess"""
    try:
        data = request.json
        session_id = data.get('session_id')
        guess_word = data.get('guess')
        
        if not session_id or not guess_word:
            return jsonify({
                "success": False,
                "message": "Missing session_id or guess"
            })
        
        result = process_guess(session_id, guess_word)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error processing guess: {str(e)}"
        })

@app.route('/api/hint', methods=['POST'])
def hint():
    """Use a hint"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                "success": False,
                "message": "Missing session_id"
            })
        
        result = use_hint(session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error using hint: {str(e)}",
            "game_over": False
        })

# Run the application
if __name__ == "__main__":
    # Initialize models and cache
    initialize_models()
    cache_dir = get_cache_directory()
    clue_cache = load_clue_cache()
    
    # Start the server
    app.run(debug=True, host='0.0.0.0', port=5000)
