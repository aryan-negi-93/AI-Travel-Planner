import os
from typing import Optional
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()



# Page configuration
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ AI Travel Planner")
st.caption("Plan trips and search real-time flight details seamlessly.")

# Initialize Model & Tools
@st.cache_resource
def get_chain():
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY")
    )

    @tool
    def search_flights(
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        departure_date: Optional[str] = None,
        return_date: Optional[str] = None
    ):
        """Search flight information from the web using TinyFish Search API."""
        if not origin or not destination or not departure_date:
            return "Missing required flight parameters (origin, destination, departure_date)."

        api_key = os.getenv("TINYFISH_API_KEY")
        if not api_key:
            return "TINYFISH_API_KEY is missing."

        if return_date:
            query = f"flights from {origin} to {destination} departing {departure_date} returning {return_date}"
        else:
            query = f"one way flights from {origin} to {destination} on {departure_date}"

        try:
            response = requests.get(
                "https://api.search.tinyfish.ai",
                headers={"X-API-Key": api_key},
                params={"query": query},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return f"TinyFish API Error: {str(e)}"

    model_tool = model.bind_tools([search_flights])

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are an expert, friendly AI Travel Planner.

You can handle normal conversations, trip planning, destination recommendations, and flight searches.

CRITICAL TOOL EXECUTION RULE:
You have access to the tool `search_flights`.
Required parameters:
1. origin
2. destination
3. departure_date

(return_date is optional)

RULES:
1. DO NOT call `search_flights` unless ALL THREE required parameters (origin, destination, departure_date) are explicitly known.
2. If ANY required parameter is missing, talk naturally to the user and ask for the missing details.
3. Extract missing parameters using previous `chat_history`.
4. City names are sufficient (e.g., Delhi, Tokyo, London).
5. Never fabricate flight information.
"""
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{query}")
    ])

    chain = prompt | model_tool
    return chain, model_tool, search_flights

chain, model_tool, search_flights = get_chain()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for resetting conversation
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat", type="primary"):
        st.session_state.chat_history = []
        st.rerun()

# Display existing messages from history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle user input
if query := st.chat_input("Ask about trips or flight schedules..."):
    # Render user query immediately
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain.invoke({
                "query": query,
                "chat_history": st.session_state.chat_history
            })

            if result.tool_calls:
                st.info("🔍 Searching flights...")
                tool_call = result.tool_calls[0]
                tool_result = search_flights.invoke(tool_call['args'])

                st.session_state.chat_history.append(HumanMessage(content=query))
                st.session_state.chat_history.append(result)
                st.session_state.chat_history.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    )
                )

                final_result = model_tool.invoke([
                    SystemMessage(
                        content="You are a friendly AI Travel Planner. Present the flight search results concisely and accurately. Do not invent details."
                    ),
                    *st.session_state.chat_history[-8:]
                ])

                response_text = final_result.content
                st.markdown(response_text)
                st.session_state.chat_history.append(final_result)

            else:
                response_text = result.content
                st.markdown(response_text)
                st.session_state.chat_history.append(HumanMessage(content=query))
                st.session_state.chat_history.append(result)
