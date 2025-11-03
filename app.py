# =============================================================
# 🔎 LangChain + Groq + Streamlit: Search & Summarization Agent
# =============================================================

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_hub import LangChainHubClient   # ✅ Correct modern import
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# -------------------------------------------------------------
# 🌍 Load environment variables
# -------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------
# 🏷️ Streamlit UI Setup
# -------------------------------------------------------------
st.set_page_config(page_title="🔎 LangChain Search Assistant", page_icon="🦜")
st.title("🦜 LangChain - Chat with Search (ReAct Agent)")
st.sidebar.title("⚙️ Settings")

# Sidebar for API key input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# -------------------------------------------------------------
# 🧰 Initialize LangChain Tools
# -------------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Web Search")

tools = [search, arxiv, wiki]

# -------------------------------------------------------------
# 💬 Session State for Conversation
# -------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm a LangChain ReAct agent that can search the web, Wikipedia, and Arxiv. How can I help you today?",
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------------------------------------
# 💭 User Input
# -------------------------------------------------------------
if user_input := st.chat_input("Ask me anything (e.g., 'What is quantum computing?')"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # ---------------------------------------------------------
    # 🧠 Initialize LLM (Groq)
    # ---------------------------------------------------------
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar before chatting.")
        st.stop()

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # ---------------------------------------------------------
    # 🧩 Load ReAct Prompt from LangChain Hub
    # ---------------------------------------------------------
    try:
        client = LangChainHubClient()
        react_prompt = client.pull("hwchase17/react")  # ✅ Pull prompt from Hub
    except Exception as e:
        st.error(f"Error loading ReAct prompt from LangChain Hub: {e}")
        st.stop()

    # ---------------------------------------------------------
    # ⚙️ Create and Execute the ReAct Agent
    # ---------------------------------------------------------
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            response = agent_executor.invoke(
                {"input": user_input},
                config={"callbacks": [st_cb]},
            )
            final_answer = response.get("output", "[No response generated]")
        except Exception as e:
            final_answer = f"⚠️ Error during processing: {e}"

        st.session_state.messages.append({"role": "assistant", "content": final_answer})
        st.write(final_answer)
