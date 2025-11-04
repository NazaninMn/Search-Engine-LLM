import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
import traceback
import requests
from bs4 import BeautifulSoup

## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Simple web search function using DuckDuckGo Lite (no API needed)
def simple_web_search(query, num_results=3):
    """Simple web search using DuckDuckGo HTML"""
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={requests.utils.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.find_all('tr')[:num_results*2]:  # Get more rows to filter
            links = result.find_all('a', class_='result-link')
            snippets = result.find_all('td', class_='result-snippet')
            
            if links and snippets:
                title = links[0].get_text(strip=True)
                snippet = snippets[0].get_text(strip=True)
                if title and snippet:
                    results.append(f"{title}: {snippet}")
        
        return "\n\n".join(results[:num_results]) if results else "No results found"
    except Exception as e:
        return f"Search error: {str(e)}"

st.title("🔎 LangChain - Chat with search")
"""
An interactive chatbot that can search the web, query ArXiv papers, and search Wikipedia using LangChain and Groq.
"""

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Add clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]
    st.rerun()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Simple tool execution function
def use_tool(tool_name, query):
    """Execute a tool based on its name"""
    try:
        if "search" in tool_name.lower() or "web" in tool_name.lower():
            return simple_web_search(query)
        elif "arxiv" in tool_name.lower():
            return arxiv.run(query)
        elif "wiki" in tool_name.lower():
            return wiki.run(query)
        else:
            return "Tool not found"
    except Exception as e:
        return f"Error using tool: {str(e)}"

# Chat input
if prompt := st.chat_input(placeholder="What is machine learning?"):
    # Check if API key is provided
    if not api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    
    # Generate response
    with st.chat_message("assistant"):
        st_container = st.container()
        
        try:
            # Create a system message explaining available tools
            system_prompt = """You are a helpful assistant with access to the following tools:
1. Search (DuckDuckGo) - for web searches
2. ArXiv - for searching academic papers
3. Wikipedia - for encyclopedia information

When you need information, think about which tool to use and tell me. I'll execute it for you.
Answer questions directly when you can, or suggest which tool to use for more information."""

            # Simple approach: Ask LLM if it needs tools
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            with st_container:
                response = llm.invoke(messages)
                answer = response.content
                
                # Check if the response suggests using a tool
                if any(keyword in answer.lower() for keyword in ["search", "arxiv", "wikipedia", "look up", "find"]):
                    st.info("🔍 Searching for information...")
                    
                    # Try to use relevant tools
                    search_results = []
                    if "arxiv" in answer.lower() or "paper" in answer.lower() or "research" in answer.lower():
                        st.write("📚 Searching ArXiv...")
                        result = use_tool("arxiv", prompt)
                        search_results.append(("ArXiv", result))
                    
                    if "wikipedia" in answer.lower() or "wiki" in answer.lower():
                        st.write("📖 Searching Wikipedia...")
                        result = use_tool("wiki", prompt)
                        search_results.append(("Wikipedia", result))
                    
                    # Default to web search
                    if not search_results or "search" in answer.lower():
                        st.write("🌐 Searching the web...")
                        result = use_tool("search", prompt)
                        search_results.append(("Web Search", result))
                    
                    # Synthesize answer with search results
                    if search_results:
                        context = "\n\n".join([f"{name}: {result[:500]}" for name, result in search_results])
                        final_messages = [
                            {"role": "system", "content": "You are a helpful assistant. Use the following search results to answer the user's question."},
                            {"role": "user", "content": f"Question: {prompt}\n\nSearch Results:\n{context}\n\nProvide a comprehensive answer based on these results."}
                        ]
                        final_response = llm.invoke(final_messages)
                        answer = final_response.content
                
                st.session_state.messages.append({'role': 'assistant', "content": answer})
                st.write(answer)
                
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}\n\n{traceback.format_exc()}"
            st.error(error_msg)
            st.session_state.messages.append({
                'role': 'assistant',
                "content": f"Sorry, I encountered an error: {str(e)}"
            })
