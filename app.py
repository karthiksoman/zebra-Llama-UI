import streamlit as st
import requests
import time

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
    response = response.split('End of response')[0].strip()
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Chat EDS with Zebra Llama")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add a sidebar for the temperature slider
with st.sidebar:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    st.caption("Sets response creativity: lower value for focused, higher value for diverse.")

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