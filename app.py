# ------------------------------------------------------------
# 🔎 LangChain + Groq + Streamlit (Stable Version for Hugging Face)
# ------------------------------------------------------------

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv

# ------------------------------------------------------------
# 🌍 Load environment variables
# ------------------------------------------------------------
load_dotenv()

st.set_page_config(page_title="Search Engine LLM", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search (Groq + Streamlit)")
st.sidebar.title("Settings")

# API key input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ------------------------------------------------------------
# 🧰 Initialize Tools
# ------------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")
tools = [search, arxiv, wiki]

# ------------------------------------------------------------
# 💬 Session state
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi 👋 I'm a chatbot that can search Arxiv, Wikipedia, and the web!"}
    ]

# Display previous chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------------------------------------------------
# 💡 User Input
# ------------------------------------------------------------
if user_input := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    if not api_key:
        st.error("Please enter your Groq API key in the sidebar before chatting.")
    else:
        # Initialize Groq LLM
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

        # Create the agent (ReAct-style)
        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=True
        )

        # Stream response
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(user_input, callbacks=[st_cb])
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
