from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from data import search_duckduckgo, fetch_stock_history, store_pdf_in_vector_db
from langchain.tools import Tool
from typing import List, Dict, Any

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

def search_tool(query: str) -> Dict[str, Any]:
    """
    Tool function to search DuckDuckGo for stock-related information.
    
    Args:
        query (str): The search query
        
    Returns:
        Dict[str, Any]: Search results
    """
    return search_duckduckgo(query)

# Create tools for the web search agent
web_search_tools = [
    Tool(
        name="search_duckduckgo",
        func=search_tool,
        description="""Search DuckDuckGo for stock-related information. 
        Returns both general search results and news articles.
        Input should be a search query string.
        Example: "AAPL stock price today" or "Tesla earnings report"."""
    )
]

# Create tools for the stock history agent
stock_history_tools = [
    Tool(
        name="fetch_stock_history",
        func=fetch_stock_history,
        description="""Fetch historical stock data. 
        Input should be a dictionary with 'ticker', 'start_date', and 'end_date' keys.
        Dates should be in 'YYYY-MM-DD' format.
        Example: {"ticker": "AAPL", "start_date": "2024-01-01", "end_date": "2024-03-01"}"""
    )
]

web_search_agent = create_react_agent(
    model=llm,
    tools=web_search_tools,
    name="DuckDuckGo Search Agent",
    prompt="""You are a DuckDuckGo Search agent specialized in stock market information. 
    Your task is to search for and provide relevant stock market information and news.
    When using the search_duckduckgo tool:
    1. Formulate a clear search query based on the user's request
    2. Use the tool to search for information
    3. Analyze both general search results and news articles
    4. Provide a comprehensive response that includes:
       - Key information from general search results
       - Recent news and updates
       - Links to sources when available
    Always format your responses in a clear and organized manner."""
)

stock_history_agent = create_react_agent(
    model=llm,
    tools=stock_history_tools,
    name="Stock History Agent",
    prompt="""You are a stock history agent specialized in providing historical stock data.
    Your task is to fetch and analyze historical stock data for users.
    When using the fetch_stock_history tool:
    1. Ensure you have all required parameters:
       - ticker: The stock symbol (e.g., 'AAPL')
       - start_date: Start date in 'YYYY-MM-DD' format
       - end_date: End date in 'YYYY-MM-DD' format
    2. Analyze the historical data
    3. Provide insights about:
       - Price trends
       - Trading volume
       - Significant changes
    Always format your responses in a clear and organized manner."""
)