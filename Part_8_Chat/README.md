# Part 8: Chat Elements in Streamlit

This section covers chat interface components that enable building conversational applications like chatbots and virtual assistants.

## Overview

Streamlit's chat elements provide a way to create interactive conversation interfaces. These are particularly useful for:
- Building AI chatbots
- Creating customer support interfaces
- Developing virtual assistants
- User feedback systems

## Files in This Section

- **[app.py](Part_8_Chat/app.py)** - Main application demonstrating basic chat functionality
- **[chat.py](Part_8_Chat/chat.py)** - Tutorial script with detailed comments and more complex examples

## Key Concepts Covered

### 1. st.chat_message()

Creates a container for displaying chat messages in the app.

```python
# Display a user message
with st.chat_message("user"):
    st.write("Hi there! How can I help you today?")

# Display a bot response
with st.chat_message("bot"):
    st.write("Sure! I can assist you with any questions.")
```

**Parameters:**
- `message_type` - The role of the message sender ("user" or "bot")
- Custom styling is applied based on the message type

**Use case:** Wraps content in a styled container that looks like a chat bubble.

---

### 2. st.chat_input()

Creates a text input field for users to type messages.

```python
prompt = st.chat_input("Type your message here...")
if prompt:
    # Process user input
    with st.chat_message("user"):
        st.write(prompt)
```

**Parameters:**
- `placeholder` - Placeholder text shown when empty

**Returns:** The user's input as a string, or None if empty

---

### 3. Building a Simple Chatbot

```python
# Set up page configuration
st.set_page_config(
    page_title="Chat App",
    page_icon="💬"
)

prompt = st.chat_input("Type your message here...")

if prompt:
    # Display user's message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate and display bot response
    with st.chat_message("bot"):
        st.write("I am a bot")
```

---

### 4. Conditional Responses

You can create more sophisticated responses based on user input:

```python
if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("bot"):
        if "help" in prompt.lower():
            st.write("It looks like you need help.")
        elif "streamlit" in prompt.lower():
            st.write("Streamlit is awesome for building data apps!")
        else:
            st.write("I'm not sure how to respond to that.")
```

## How It Works

1. **Chat Input**: User types a message using `st.chat_input()`
2. **Message Display**: Messages are displayed using `st.chat_message()` with appropriate styling
3. **Response Generation**: The app processes the input and generates a response (could integrate with LLM APIs)
4. **State Management**: Use session state to maintain conversation history

## Advanced: Maintaining Chat History

To build a full chatbot, you'll need to store messages in session state:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle new input
if prompt := st.chat_input("Type your message..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response (example)
    response = f"You said: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
```

## Running the App

```bash
streamlit run Part_8_Chat/app.py
# or
streamlit run Part_8_Chat/chat.py
```

## Next Steps

After completing this section, you should understand:
- How to create chat message containers with different styles
- Building a basic chatbot interface
- Handling user input and generating responses

Proceed to [Part 9: Status](../Part_9_Status/) to learn about status indicators and progress displays.