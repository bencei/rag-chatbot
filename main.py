from backend.core import qa
import streamlit as st
from streamlit_chat import message

st.header("Raffi wiki helper bot")

prompt = st.text_input("Prompt", placeholder="Ask me anything...")

if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

if prompt:
    with st.spinner("Thinking..."):
        response = qa(query=prompt, chat_history=st.session_state["chat_history"])
        # sources = set([doc.metadata["source"] for doc in response["source_documents"]])
        formatted_response = f"**Answer:** {response['result']}\n\n"
        print(f"Formatted Response: {formatted_response}")
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", response["result"]))

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)
        message(generated_response)
