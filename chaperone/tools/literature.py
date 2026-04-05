from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Create default instances of community tools
arxiv_searcher = ArxivQueryRun()
ddg_searcher = DuckDuckGoSearchRun()

@tool
def search_literature(query: str) -> str:
    """
    Search ArXiv for preprints related to biology, protein engineering, deep learning, etc.
    Use this to look up specific papers or recent advancements not in your local RAG database.
    """
    return arxiv_searcher.run(query)

@tool
def web_search(query: str) -> str:
    """
    Query the internet via DuckDuckGo for documentation, tutorials, or open source repositories.
    Use this when you need to research an external API, look up RFDiffusion or ProteinMPNN syntax, 
    or check biological databases online.
    """
    return ddg_searcher.run(query)
