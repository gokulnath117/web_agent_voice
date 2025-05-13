from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph_supervisor import create_supervisor
from agent import web_search_agent, stock_history_agent
import json
from datetime import datetime, timedelta
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

prompt = """You are a stock market assistant that coordinates between different specialized agents.
Your role is to:
1. Understand the user's request
2. Determine which agent(s) should handle the request
3. Process and present the results in a clear, organized manner

Available agents:
- Google Search Agent: For finding stock-related news and information
- Stock History Agent: For fetching historical stock data

Always ensure the response is well-formatted and includes all relevant information."""

supervisor = create_supervisor(
    [web_search_agent, stock_history_agent],
    model=llm,
    output_mode="full_history",
    prompt=prompt,
)

app = supervisor.compile()

def process_user_query(query: str):
    """
    Process a user query through the supervisor and agents.
    
    Args:
        query (str): The user's query
        
    Returns:
        dict: The processed response
    """
    # Prepare the message
    message = HumanMessage(content=query)
    
    # Invoke the supervisor
    result = app.invoke({
        "messages": [message]
    })
    
    # Process and return the result
    return result

# Example usage
if __name__ == "__main__":
    # Example query
    query = "What is the stock price of AAPL and what are the recent news about it?"
    
    # Get the response
    response = process_user_query(query)
    
    # Print the response in a formatted way
    print("\n=== Response ===")
    for message in response['messages']:
        if isinstance(message, AIMessage):
            print(f"\nAssistant: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"\nUser: {message.content}")