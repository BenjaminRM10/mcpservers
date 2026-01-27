import os
import requests
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient

# Initialize FastMCP server
mcp = FastMCP("tavily-search")

def get_api_key() -> str:
    """Helper to get the Tavily API key from environment."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set.")
    return api_key

def get_client() -> TavilyClient:
    """Helper to get the Tavily client."""
    return TavilyClient(api_key=get_api_key())

@mcp.tool()
async def tavily_search(
    query: str,
    search_depth: str = "basic",
    topic: str = "general",
    max_results: int = 5,
    include_answer: bool = False,
    include_raw_content: bool = False,
    include_images: bool = False,
    include_domains: List[str] = [],
    exclude_domains: List[str] = [],
) -> str:
    """
    Execute a search query using Tavily Search.

    Args:
        query: The search query to execute.
        search_depth: "basic" (balanced) or "advanced" (high relevance, higher cost).
        topic: "general" or "news".
        max_results: Maximum number of search results to return.
        include_answer: If True, include an LLM-generated answer.
        include_raw_content: If True, include cleaned and parsed HTML content.
        include_images: If True, include relevant images.
        include_domains: List of domains to specifically include.
        exclude_domains: List of domains to specifically exclude.
    """
    try:
        client = get_client()
        response = client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            include_images=include_images,
            include_domains=include_domains or None,
            exclude_domains=exclude_domains or None
        )
        return str(response)
    except Exception as e:
        return f"Tavily Search failed: {str(e)}"

@mcp.tool()
async def tavily_extract(urls: List[str]) -> str:
    """
    Extract web page content from specified URLs.

    Args:
        urls: List of URLs to extract content from.
    """
    try:
        # Using raw request since SDK might vary on this implementation
        api_key = get_api_key()
        response = requests.post(
            "https://api.tavily.com/extract",
            json={"urls": urls, "api_key": api_key},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Tavily Extract failed: {str(e)}"

@mcp.tool()
async def tavily_crawl(url: str, max_depth: int = 1, limit: int = 10) -> str:
    """
    Crawl a website to discover and extract content.

    Args:
        url: The root URL to begin the crawl.
        max_depth: How far from the base URL to explore.
        limit: Total number of links to process.
    """
    try:
        api_key = get_api_key()
        response = requests.post(
            "https://api.tavily.com/crawl",
            json={
                "url": url, 
                "api_key": api_key,
                "max_depth": max_depth,
                "limit": limit
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Tavily Crawl failed: {str(e)}"

@mcp.tool()
async def tavily_map(url: str, max_depth: int = 1, max_breadth: int = 10) -> str:
    """
    Generate a sitemap by traversing a website.

    Args:
        url: The root URL to map.
        max_depth: Depth of the mapping.
        max_breadth: Number of links to follow per level.
    """
    try:
        api_key = get_api_key()
        response = requests.post(
            "https://api.tavily.com/map",
            json={
                "url": url, 
                "api_key": api_key,
                "max_depth": max_depth,
                "max_breadth": max_breadth
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Tavily Map failed: {str(e)}"

@mcp.tool()
async def tavily_research(topic: str, model: str = "auto") -> str:
    """
    Perform comprehensive research on a topic.

    Args:
        topic: The research task or question.
        model: "mini", "pro", or "auto".
    """
    try:
        api_key = get_api_key()
        response = requests.post(
            "https://api.tavily.com/research",
            json={
                "input": topic, 
                "api_key": api_key,
                "model": model
            },
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Tavily Research failed: {str(e)}"

@mcp.tool()
async def tavily_usage() -> str:
    """
    Check current API key usage and limits.
    """
    try:
        api_key = get_api_key()
        # GET request requires Authorization header for this endpoint per docs
        response = requests.get(
            "https://api.tavily.com/usage",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        # Fallback to key param if Auth header fails (some endpoints vary)
        try:
             response = requests.get(
                f"https://api.tavily.com/usage?api_key={api_key}"
            )
             response.raise_for_status()
             return str(response.json())
        except:
            return f"Tavily Usage failed: {str(e)}"

if __name__ == "__main__":
    mcp.run()
