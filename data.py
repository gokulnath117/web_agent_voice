import requests
from bs4 import BeautifulSoup
import yfinance as yf
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from typing import Dict, Any, List
from duckduckgo_search import DDGS
from newspaper import Article
from textblob import TextBlob
import re

def extract_keywords(text: str) -> List[str]:
    """
    Extract keywords from text using TextBlob.
    """
    if not text:
        return []
    
    # Get noun phrases and important words
    blob = TextBlob(text)
    keywords = []
    
    # Add noun phrases
    keywords.extend([phrase for phrase in blob.noun_phrases])
    
    # Add important words (nouns, verbs, adjectives)
    for word, tag in blob.tags:
        if tag.startswith(('NN', 'VB', 'JJ')) and len(word) > 3:
            keywords.append(word)
    
    return list(set(keywords))

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze sentiment of text using TextBlob.
    """
    if not text:
        return {"sentiment": "neutral", "score": 0}
    
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    
    if score > 0.1:
        sentiment = "positive"
    elif score < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "sentiment": sentiment,
        "score": score
    }

def get_news_summary(url: str) -> Dict[str, Any]:
    """
    Scrapes article title and summary using newspaper3k.
    
    Args:
        url (str): URL of the news article
        
    Returns:
        Dict[str, Any]: Article title and summary
    """
    try:
        article = Article(url)
        article.download()
        article.parse()

        return {
            "title": article.title,
            "summary": article.meta_description if article.meta_description else article.text[:250],
            "source": url.split('/')[2],
            "url": url
        }
    except Exception as e:
        print(f"Error extracting {url}: {e}")
        return None

def get_news_urls(query: str, max_results: int = 5) -> List[str]:
    """
    Fetches news article URLs related to a given query from Google News.
    
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return
        
    Returns:
        List[str]: List of news article URLs
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}+news&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        remove_url = [
            'https://www.google.com',
            'https://maps.google.com',
            'https://play.google.com',
            'https://policies.google.com',
            'https://support.google.com',
            'https://accounts.google'
        ]

        news_links = set()
        for link in soup.find_all('a', href=True):
            url = link['href']
            if '/url?q=' in url:
                filter_url = url.split("/url?q=")[-1].split("&")[0]
                if ("http" in filter_url and 
                    'msn' not in filter_url and 
                    not any(remove in filter_url for remove in remove_url)):
                    news_links.add(filter_url)
                    if len(news_links) >= max_results:
                        break
        
        return list(news_links)
    except Exception as e:
        print(f"Error fetching news URLs: {e}")
        return []

def search_duckduckgo(query: str) -> Dict[str, Any]:
    """
    Search DuckDuckGo and Google News for relevant information.
    
    Args:
        query (str): The search query
        
    Returns:
        Dict[str, Any]: A dictionary containing the search results and status
    """
    try:
        # Get DuckDuckGo results
        with DDGS() as ddgs:
            text_results = list(ddgs.text(query, max_results=5))
            news_results = list(ddgs.news(query, max_results=3))
        
        # Get Google News URLs and summaries
        news_urls = get_news_urls(query, max_results=5)
        news_summaries = []
        for url in news_urls:
            summary = get_news_summary(url)
            if summary:
                news_summaries.append(summary)
        
        # Combine and format results
        results = {
            "text_results": [
                {
                    "title": result["title"],
                    "snippet": result["body"],
                    "link": result["link"]
                }
                for result in text_results
            ],
            "news_results": [
                {
                    "title": result["title"],
                    "snippet": result["body"],
                    "link": result["link"],
                    "date": result.get("date", "")
                }
                for result in news_results
            ],
            "news_summaries": news_summaries
        }
        
        return {
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "results": None
        }

def fetch_stock_history(ticker: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Fetches historical stock data for a given ticker symbol and date range.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        
    Returns:
        Dict[str, Any]: A dictionary containing the stock data and status
    """
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(start=start_date, end=end_date)
        if history.empty:
            return {"status": "error", "message": "No data found for the given period.", "data": None}
        
        # Convert DataFrame to dictionary for better serialization
        data = history.to_dict('records')
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e), "data": None}

def store_pdf_in_vector_db(pdf_path: str) -> Dict[str, Any]:
    """
    Stores a PDF document in a FAISS vector database for semantic search.
    
    Args:
        pdf_path (str): Path to the PDF file to be stored
        
    Returns:
        Dict[str, Any]: A dictionary containing the status and vector database info
    """
    try:
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Generate embeddings
        embeddings = OpenAIEmbeddings()

        # Store in FAISS vector database
        vector_db = FAISS.from_documents(documents, embeddings)
        return {"status": "success", "message": "PDF successfully stored in the vector database."}
    except Exception as e:
        return {"status": "error", "message": str(e)}