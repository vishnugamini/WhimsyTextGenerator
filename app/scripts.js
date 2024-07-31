document.getElementById('sendButton').addEventListener('click', sendPrompt);
document.getElementById('stopButton').addEventListener('click', stopGenerating);

let controller;

async function sendPrompt() {
    const initialText = document.getElementById('initialText').value;
    const maxLength = document.getElementById('maxLength').value;
    const chatWindow = document.getElementById('chatWindow');

    if (initialText.trim() === '') {
        showNotification('Please enter a prompt.');
        return;
    }

    const userMessage = createMessageElement(initialText, 'user');
    chatWindow.appendChild(createMessageContainer(userMessage, null));
    chatWindow.scrollTop = chatWindow.scrollHeight;

    document.getElementById('initialText').value = '';

    const responseMessage = createMessageElement('', 'ai');
    const responseContainer = createMessageContainer(responseMessage, createCopyGlyph(''));
    chatWindow.appendChild(responseContainer);
    chatWindow.scrollTop = chatWindow.scrollHeight;

    controller = new AbortController();
    const { signal } = controller;

    document.getElementById('stopButton').disabled = false;

    try {
        const response = await fetch('http://localhost:5000/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ initial_text: initialText, max_length: maxLength }),
            signal
        });

        if (response.ok) {
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            responseMessage.textContent = ''; // Clear previous content

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const textChunk = decoder.decode(value, { stream: true });
                responseMessage.innerHTML += textChunk.replace(/\n/g, '<br>'); // Preserve line breaks
                responseContainer.querySelector('.copy-glyph').dataset.text = responseMessage.innerHTML;
                chatWindow.scrollTop = chatWindow.scrollHeight;
            }
        } else {
            responseMessage.textContent = 'Error generating text.';
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            responseMessage.textContent = 'Text generation stopped.';
        } else {
            responseMessage.textContent = 'Error generating text.';
        }
    } finally {
        document.getElementById('stopButton').disabled = true;
    }
}

function stopGenerating() {
    if (controller) {
        controller.abort();
    }
}

function createMessageContainer(messageElement, copyGlyph) {
    const container = document.createElement('div');
    container.className = 'message-container';
    container.appendChild(messageElement);
    if (copyGlyph) container.appendChild(copyGlyph);
    return container;
}

function createMessageElement(text, sender) {
    const message = document.createElement('div');
    message.className = `message ${sender}`;
    message.innerHTML = text.replace(/\n/g, '<br>'); // Preserve line breaks
    return message;
}

function createCopyGlyph(text) {
    const glyph = document.createElement('div');
    glyph.className = 'copy-glyph';
    glyph.dataset.text = text;
    glyph.addEventListener('click', () => copyToClipboard(glyph.dataset.text));
    return glyph;
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Text copied to clipboard');
    }).catch(err => {
        showNotification('Failed to copy text');
    });
}

function showNotification(message) {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.classList.add('show');

    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}
