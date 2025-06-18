const homeSection = document.getElementById('home-section');
const characterSection = document.getElementById('character-section');
const chatbotSection = document.getElementById('chatbot-section');
const chatbotTitle = document.getElementById('chatbot-title');
const topicContainer = document.getElementById('topic-container');
const topicInput = document.getElementById('topic');
const dialogueCharacterSelect = document.getElementById('dialogue-character');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const chatArea = document.getElementById('chat-area');
const chatContainer = document.querySelector('.chat-container');
const homeBtn = document.getElementById('home-btn');
const characterCards = document.querySelectorAll('.character-card');

let chatHistory = [];
let currentCharacter = null;

// Show Home Section by Default
function showHome() {
    homeSection.classList.remove('hidden');
    characterSection.classList.remove('hidden');
    chatbotSection.classList.add('hidden');
    chatArea.innerHTML = ''; // Clear chat
    chatHistory = [];
    userInput.value = '';
    topicInput.value = '';
    currentCharacter = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show Chatbot Section with Selected Character
function showChatbot(character) {
    homeSection.classList.add('hidden');
    characterSection.classList.add('hidden');
    chatbotSection.classList.remove('hidden');
    currentCharacter = character;
    chatbotTitle.textContent = character === 'generate' ? 'Generate New Dialogue' : character === 'None' ? 'General Chat' : `Chatting with ${character}`;
    if (character === 'generate') {
        topicContainer.classList.remove('hidden');
        userInput.parentElement.classList.add('hidden');
    } else {
        topicContainer.classList.add('hidden');
        userInput.parentElement.classList.remove('hidden');
    }
}

// Handle Character Card Click
characterCards.forEach(card => {
    card.addEventListener('click', () => {
        const character = card.getAttribute('data-character');
        showChatbot(character);
    });
});

// Handle Home Button Click
homeBtn.addEventListener('click', showHome);

// Send Message or Generate Dialogue
sendBtn.addEventListener('click', async () => {
    const character = currentCharacter;
    const userText = userInput.value.trim();
    const topic = topicInput.value.trim();
    const dialogueCharacter = dialogueCharacterSelect.value;

    if (character === 'generate' && !topic) {
        appendMessage('System', 'Please enter a topic for the dialogue!', 'text-red-600');
        return;
    }

    if (character !== 'generate' && !userText) {
        appendMessage('System', 'Please enter a message!', 'text-red-600');
        return;
    }

    appendMessage('You', userText || topic, 'text-blue-600');

    try {
        let response;
        if (character === 'generate') {
            response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    topic,
                    character: dialogueCharacter === 'None' ? null : dialogueCharacter
                })
            });
        } else {
            response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_input: userText,
                    character: character === 'None' ? null : character,
                    chat_history: chatHistory.slice(-3)
                })
            });
        }

        const data = await response.json();
        if (response.ok) {
            const botResponse = data.response || 'Hmm, Central Perk’s out of coffee! Try again!';
            appendMessage(character === 'None' ? 'Bot' : character, botResponse, 'text-purple-600');

            // Update chat history for chat mode
            if (character !== 'generate') {
                chatHistory.push({ user: userText, bot: botResponse, character });
                chatHistory = chatHistory.slice(-3);
            }

            // Reset input for chat mode
            if (character !== 'generate') {
                userInput.value = '';
            }
        } else {
            appendMessage('System', data.response || 'Error occurred!', 'text-red-600');
        }

        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (error) {
        appendMessage('System', 'Error connecting to Central Perk! Try again later.', 'text-red-600');
        console.error(error);
    }
});

// Append Message to Chat Area
function appendMessage(sender, message, colorClass) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `p-2 rounded-lg max-w-[80%] ${sender === 'You' ? 'bg-blue-100 ml-auto' : 'bg-purple-100'} ${colorClass}`;
    messageDiv.innerHTML = `<strong>${sender}:</strong> ${message.replace(/\n/g, '<br>')}`;
    chatArea.appendChild(messageDiv);
}

// Handle Enter Key for Sending Messages
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});

// Initialize Home View
showHome();