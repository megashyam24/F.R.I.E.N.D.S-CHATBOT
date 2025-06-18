from flask import Flask, render_template, request, jsonify
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import re

app = Flask(__name__, template_folder='templates')

# Configure Gemini API
api_key = "API_KEY"
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Define character personas
character_personas = {
    'Chandler': 'sarcastic, witty, uses humor to deflect, often says "Could I *be* more..."',
    'Joey': 'goofy, charming, loves food and women, says "How you doin\'?"',
    'Monica': 'controlling, competitive, meticulous, loves cleaning',
    'Ross': 'nerdy, sensitive, often whiny, obsessed with dinosaurs',
    'Rachel': 'fashionable, charming, sometimes ditzy, loves shopping',
    'Phoebe': 'quirky, spiritual, eccentric, loves singing odd songs'
}

# Load and preprocess dataset
def load_dataset(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8').dropna()
        main_characters = ['Chandler', 'Joey', 'Monica', 'Ross', 'Rachel', 'Phoebe']
        df = df[df['Name'].isin(main_characters)]
        if df.empty:
            raise ValueError("No data found for main characters in the dataset.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

# Set up vector database with chunking and batch processing
def setup_vector_db(df, persist_directory="./chroma_db", chunk_size=5, batch_size=5000):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="friends_dialogue")
    
    if collection.count() == 0:  # Only add data if collection is empty
        print("Setting up new ChromaDB collection...")
        lines = df['Lines'].tolist()
        characters = df['Name'].tolist()
        
        # Create chunks of consecutive lines
        chunks = []
        chunk_metadata = []
        chunk_ids = []
        chunk_counter = 0
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_chars = characters[i:i + chunk_size]
            chunk_text = "\n".join([f"{char}: {line}" for char, line in zip(chunk_chars, chunk_lines)])
            
            # Add chunk for each unique character in it
            for char in set(chunk_chars):
                chunks.append(chunk_text)
                chunk_metadata.append({"character": char})
                chunk_ids.append(f"chunk_{chunk_counter}_{char}")
                chunk_counter += 1
        
        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_metadata = chunk_metadata[i:i + batch_size]
            batch_ids = chunk_ids[i:i + batch_size]
            batch_embeddings = embedder.encode(batch_chunks, convert_to_tensor=False)
            
            try:
                collection.add(
                    documents=batch_chunks,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                print(f"Added batch {i // batch_size + 1} of {(len(chunks) + batch_size - 1) // batch_size}")
            except Exception as e:
                print(f"Error adding batch {i // batch_size + 1}: {str(e)}")
                raise
    else:
        print("Loaded existing ChromaDB collection.")
    
    return collection

# Retrieve relevant chunks
def retrieve_lines(user_input, character, collection, k=3):
    try:
        input_embedding = embedder.encode([user_input], convert_to_tensor=False)[0]
        query_params = {"query_embeddings": [input_embedding], "n_results": k}
        if character:  # Only filter by character if one is specified
            query_params["where"] = {"character": {"$eq": character}}
        results = collection.query(**query_params)
        
        # Ensure documents and metadata are not empty
        documents = results.get('documents', [[]])[0] or ["No relevant dialogue found, but I can still chat about Friends!"]
        metadatas = results.get('metadatas', [[]])[0] or [{"character": "None"} for _ in documents]
        
        # Ensure metadata matches documents in length
        if len(metadatas) < len(documents):
            metadatas.extend([{"character": "None"}] * (len(documents) - len(metadatas)))
        elif len(documents) < len(metadatas):
            documents.extend(["No relevant dialogue found, but I can still chat about Friends!"] * (len(metadatas) - len(documents)))
        
        return documents, metadatas
    except Exception as e:
        print(f"Error retrieving lines: {str(e)}")
        return ["Error retrieving dialogue, but I’m ready to talk Friends!"], [{"character": "None"}]

# Detect if input references a specific Friends scene
def is_scene_query(user_input):
    scene_keywords = ['remember', 'scene', 'moment', 'episode', 'when', 'propose', 'wedding', 'kiss', 'break', 'lobster']
    return any(keyword in user_input.lower() for keyword in scene_keywords)

# Detect if input requests a long response
def is_long_response_query(user_input):
    long_keywords = ['long', 'detailed', 'passage', 'explain', 'elaborate']
    return any(keyword in user_input.lower() for keyword in long_keywords)

# Create prompt for Gemini
def create_prompt(user_input, character, retrieved_lines, retrieved_metadata, chat_history):
    base_persona = (
        "You are a joyful, witty chatbot with encyclopedic knowledge of the Friends TV show, acting like a best friend who’s watched every episode. "
        "Respond with enthusiasm, humor, and engagement, as if chatting at Central Perk. Use the provided dialogue chunks to ground your response in the show’s context, avoiding hallucination. "
        "Use plain text without Markdown formatting (no **, *, #, etc.). Keep responses to 2-3 lines (50-100 words) unless the user requests a long or detailed response."
    )
    
    persona = base_persona
    if character:
        persona += f" You are responding as {character}, whose personality is {character_personas.get(character, 'unknown')}. Mimic their tone and style."
    else:
        persona += " You are a general Friends enthusiast, using a lively, humorous tone inspired by the show’s group dynamic (think Central Perk banter). Engage the user like a friend who knows every episode, without mimicking a specific character."
    
    # Format chat history
    history_context = ""
    if chat_history:
        history_context = "Recent conversation:\n" + "\n".join(
            [f"User: {h['user']}\n{h['character'] or 'Bot'}: {h['bot']}" for h in chat_history]
        ) + "\n\n"
    
    # Format retrieved chunks
    line_context = ""
    for i, (chunk, metadata) in enumerate(zip(retrieved_lines, retrieved_metadata)):
        line_context += f"Chunk {i+1} (Character: {metadata.get('character', 'None')}):\n{chunk}\n\n"
    
    # Adjust prompt for scene queries and response length
    length_instruction = "Keep the response to 2-3 lines (50-100 words) for brevity." if not is_long_response_query(user_input) else "Provide a detailed response (up to 300 words) as requested."
    
    if is_scene_query(user_input):
        prompt = f"""
        {persona}
        {history_context}
        The user is referencing a specific moment from Friends: "{user_input}"
        Here are relevant dialogue chunks from the show:
        {line_context}
        Respond with an engaging, enthusiastic description of the scene or moment in plain text, using your Friends knowledge to fill in details (e.g., setting, emotions, iconic lines). 
        If a character is selected, weave in their style. For general chat, use a fun, Friends-group vibe, referencing the chunks for authenticity. 
        If no relevant chunks are found, rely on your Friends knowledge to provide an accurate response, keeping it lively and true to the show. 
        {length_instruction}
        Example: For "Chandler proposes to Monica," describe the proposal in Monica’s apartment with Chandler’s nervousness and lines like "You make me happier than I ever thought I could be."
        """
    else:
        prompt = f"""
        {persona}
        {history_context}
        A user says: "{user_input}"
        Here are relevant dialogue chunks from the show:
        {line_context}
        Respond in a conversational, joyful way in plain text, as if chatting with a friend who loves Friends. If a character is selected, match their style (e.g., Chandler’s sarcasm for 'I have a girlfriend' might be 'Could I be more single?'). 
        For general chat, use a lively, humorous tone inspired by the Friends group dynamic, referencing the chunks for authenticity. If no relevant chunks are found, use your Friends knowledge to keep the conversation fun and relevant. 
        {length_instruction}
        Example: For "hi," respond with something like "Hey, it’s like we’re chilling at Central Perk! What’s the vibe today—Ross’s dinosaur facts or Joey’s sandwich obsession?"
        """
    
    return prompt

# Generate response using Gemini
def generate_response(user_input, character, collection, chat_history):
    try:
        retrieved_lines, retrieved_metadata = retrieve_lines(user_input, character, collection)
        prompt = create_prompt(user_input, character, retrieved_lines, retrieved_metadata, chat_history)
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "Hmm, Central Perk’s out of coffee! Tell me more about what you love in Friends!"
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return f"Oops, looks like we’re stuck in a Friends rerun glitch! Let’s keep the Central Perk vibes going—what’s your favorite moment from the show?"

# Generate a new Friends-style dialogue
def generate_new_dialogue(character, collection, topic="Central Perk conversation"):
    try:
        retrieved_lines, retrieved_metadata = retrieve_lines(topic, character, collection, k=3)
        line_context = "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(retrieved_lines)])
        
        prompt = f"""
        You are a writer for the Friends TV show, creating a new dialogue in the style of the show.
        {f"You are writing for {character}, whose personality is {character_personas.get(character, 'unknown')}. Mimic their tone and style." if character else "Write a general Friends dialogue with the main characters."}
        Here are sample dialogue chunks for inspiration:
        {line_context}
        Create a short, humorous dialogue set in Central Perk about "{topic}". Keep it authentic to the Friends universe.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating dialogue: {str(e)}"

# Interactive chat function with continuous chat mode
def run_chatbot(collection):
    print("Welcome to the Friends Chatbot! I'm your best friend who knows *everything* about Friends!")
    print("Choose a character (Chandler, Joey, Monica, Ross, Rachel, Phoebe) or type 'None' for general chat.")
    print("Type 'generate' to create a new dialogue, or 'exit' to quit.")
    
    while True:
        try:
            character = input("\nCharacter (or None/generate/exit): ").capitalize()
            if character.lower() == 'exit':
                print("See you at Central Perk!")
                break
            elif character.lower() == 'generate':
                topic = input("Topic for new dialogue (e.g., 'Central Perk conversation'): ")
                char_for_dialogue = input("Character for dialogue (or None): ").capitalize()
                char_for_dialogue = char_for_dialogue if char_for_dialogue != 'None' else None
                dialogue = generate_new_dialogue(char_for_dialogue, collection, topic)
                print(f"\n**New Friends Dialogue**:\n{dialogue}")
                continue
            elif character not in ['None', 'Chandler', 'Joey', 'Monica', 'Ross', 'Rachel', 'Phoebe']:
                print("Invalid character! Choose from Chandler, Joey, Monica, Ross, Rachel, Phoebe, or None.")
                continue
            
            character = None if character == 'None' else character
            chat_history = []
            
            # Continuous chat mode
            print(f"\nChatting with {character or 'Bot'}. Say 'bye' to return to character selection.")
            while True:
                user_input = input(f"You to {character or 'Bot'}: ")
                if user_input.lower() in ['bye', 'goodbye', 'see ya', 'later']:
                    print(f"{character or 'Bot'}: Catch you later at Central Perk!")
                    break
                if not user_input.strip():
                    print("Please enter a valid input.")
                    continue
                
                response = generate_response(user_input, character, collection, chat_history)
                print(f"{character or 'Bot'}: {response}")
                
                # Store chat history (last 3 exchanges)
                chat_history.append({"user": user_input, "bot": response, "character": character})
                chat_history = chat_history[-3:]  # Keep only the last 3 for context
                
        except Exception as e:
            print(f"Error: {str(e)}")

# Load dataset and setup vector DB globally
try:
    df = load_dataset('Friends_script.csv')
    collection = setup_vector_db(df, persist_directory='chroma_db')
except Exception as e:
    print(f"Startup error: {str(e)}")
    collection = None

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not collection:
        return jsonify({'response': 'Error: Database not initialized!'}), 500
    data = request.json
    user_input = data.get('user_input')
    character = data.get('character')
    chat_history = data.get('chat_history', [])
    if not user_input:
        return jsonify({'response': 'Please enter a message!'}), 400
    response = generate_response(user_input, character, collection, chat_history)
    return jsonify({'response': response})

@app.route('/generate', methods=['POST'])
def generate():
    if not collection:
        return jsonify({'response': 'Error: Database not initialized!'}), 500
    data = request.json
    topic = data.get('topic')
    character = data.get('character')
    if not topic:
        return jsonify({'response': 'Please enter a topic!'}), 400
    dialogue = generate_new_dialogue(character, collection, topic)
    return jsonify({'response': dialogue})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
