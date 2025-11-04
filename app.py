import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os

load_dotenv()

st.title("🔎 LangChain - Chat with Search")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# ---- Tools ----
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun(name="Search")
tools = [search, arxiv, wiki]

# ---- Session ----
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant",
         "content": "Hi, I'm a chatbot that can search Arxiv, Wikipedia, and the web!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ---- Chat ----
if user_input := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant with access to the following tools: search, arxiv, and wikipedia.\n"
        "Use them if needed before answering.\n\nQuestion: {input}"
    )

    # ✅ Uses the stable zero-shot ReAct agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = agent.run(user_input, callbacks=[st_cb])
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)
