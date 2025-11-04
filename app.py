import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain.agents.react.agent import create_react_agent   # ✅ correct new path
from langchain.agents import AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

# ------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------
load_dotenv()

st.title("🔎 LangChain - Chat with Search (ReAct Agent)")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ------------------------------------------------------------
# Initialize tools
# ------------------------------------------------------------
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")
tools = [search, arxiv, wiki]

# ------------------------------------------------------------
# Session State
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot that can search Arxiv, Wikipedia, and the web!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ------------------------------------------------------------
# Chat Input
# ------------------------------------------------------------
if user_input := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    # Prompt (manual ReAct-style)
    react_prompt = PromptTemplate.from_template(
        "You are a helpful AI assistant that uses external tools when needed.\n"
        "Use reasoning before answering and be accurate.\n\n"
        "Question: {input}"
    )

    # ✅ Correct ReAct agent construction
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent_executor.invoke({"input": user_input}, config={"callbacks": [st_cb]})
        final_answer = response.get("output", "[No response generated]")
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        st.write(final_answer)
