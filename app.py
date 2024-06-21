import streamlit as st
import requests
import time
from PIL import Image

def get_response(user_input, temperature):
    url = "https://jfn1so7p7a.execute-api.us-west-1.amazonaws.com/Prod/inference"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "text": user_input,
        "temperature": temperature
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def response_generator(user_input, temperature):
    response = get_response(user_input, temperature)
    if not response:
        response = 'Sorry, this request cannot be processed due to token limit. You can report this issue at: https://github.com/karthiksoman/zebra-Llama/issues'
    if not isinstance(response, str):
        response = 'Sorry, encountered a glitch! Can you try it again? If you hit this same error, you can report this issue at: https://github.com/karthiksoman/zebra-Llama/issues'
    # response = response.split('End of response')[0].strip()
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

logo = Image.open("ZebraLLAMA_logo.png")

st.title("Chat EDS with Zebra Llama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a sidebar for the temperature slider
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <a href="https://github.com/karthiksoman/zebra-Llama" target="_blank" style="text-decoration: none; color: inherit; display: inline-flex; align-items: center;">
            Zebra-Llama Github
            <svg height="20" width="20" viewBox="0 0 16 16" version="1.1" style="margin-left: 5px;">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(logo, width=200, use_column_width=True)
    st.write("")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    st.caption("Sets response creativity: lower value for focused, higher value for diverse.")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("---")
    st.markdown(
        "<p style='font-size: small; color: gray;'>"
        "<strong>Disclaimer:</strong> Zebra-LLAMA is an AI model "
        "designed for academic research and educational purposes only. "
        "It should not be used as a substitute for professional medical advice, diagnosis, or treatment."
        "</p>", 
        unsafe_allow_html=True
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Chat about EDS"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Wait for ~10 seconds. Processing... "):
        with st.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt, temperature))

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

