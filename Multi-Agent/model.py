import os
from typing import Optional
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain.tools import tool
from dotenv import load_dotenv
import requests

from langchain_core.messages import AIMessage , HumanMessage , SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder

from langchain_core.output_parsers import StrOutputParser


load_dotenv()


# -------------------
# for save the chat history
# -------------------


chat_history = []

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)



# -------------------
# create tools
# -------------------

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



    print("\n🔥 [TOOL EXECUTING] SEARCH_FLIGHTS STARTED...")

    api_key = os.getenv("TINYFISH_API_KEY")
    if not api_key:
        return "TINYFISH_API_KEY is missing."

    if return_date:
        query = (
            f"flights from {origin} to {destination} "
            f"departing {departure_date} returning {return_date}"
        )
    else:
        query = (
            f"one way flights from {origin} to {destination} "
            f"on {departure_date}"
        )

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
            The required parameters are:
            1. origin
            2. destination
            3. departure_date

            (return_date is optional)

            RULES:
            1. DO NOT call `search_flights` unless ALL THREE required parameters (origin, destination, departure_date) are explicitly known.
            2. If ANY required parameter is missing, talk naturally to the user and ask for the missing details.
            3. Extract missing parameters using previous `chat_history`.
            4. City names are sufficient (e.g., Delhi, Tokyo, London). Do not force specific airport selection unless needed.
            5. Never fabricate flight information.

        """
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{query}")
])





chain = prompt | model_tool


while True:
    query = input("you: ")

    if query == "exit":
        break

    result = chain.invoke({
        "query" : query,
        "chat_history":chat_history
    })

    if result.tool_calls:
        print("========= model call the tool ==========")
        tool_Call = result.tool_calls[0]

        tool_result = search_flights.invoke(tool_Call['args'])

        chat_history.append(HumanMessage(content=query))
        chat_history.append(result)
        chat_history.append(
            ToolMessage(
                content=tool_result,
                tool_call_id= (tool_Call["id"])
            )
        )

        final_result = model_tool.invoke([
            SystemMessage(content="You are a friendly AI Travel Planner. Present the flight search results concisely and accurately. Do not invent details."),
            *chat_history[-8:]

        ])

        print(f"\nAI: {final_result.content}")
        chat_history.append(final_result)

    else:

        print(f"\nAI: {result.content}")
        chat_history.append(HumanMessage(content=query))
        chat_history.append(result)

            
        









