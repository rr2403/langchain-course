import asyncio
import os
import ssl
from typing import Any, Dict, List
from dotenv import load_dotenv
import certifi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)
load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUEST_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10
)

#chroma = Chroma(persist_directory = "chroma_db", embedding_function = embeddings)
vectorstore = PineconeVectorStore(index_name="documentation-helper", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=100, max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int=20) -> List[List[str]]:
    """Split URLs into chunks of specified size"""
    chunks=[]
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str],  batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs"""
    try:
        log_info(
            f"Tavily Extract: Processing batch {batch_num} with {len(urls)} URLs"
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(
            f" Tavily Extract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents"

        )
        return docs
    except Exception as e:
        log_error(f"Tavily Extract: Failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"Tavily Extract: Starting concurrent extraction of {len(url_batches)}",
        Colors.DARKCYAN
    )
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"Tavily Extract: Batch failed with exception - {result}")
            failed_batches+=1
        else:
            for extracted_page in result["results"]:
                document=Document(
                    page_content=extracted_page["raw_content"],
                    metadata={"source": extracted_page["url"]}
                )
                all_pages.append(document)
    log_success(
        f"TavilyExtract: Extraction complete! Total pages extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")
    return all_pages
                
async def main():
    """Main async function to orchestrate the entire process"""
    log_header("Document Ingestion Pipeline")
    log_info(
        "Tavily Map: Starting to map documentation structure from https://python.langchain.com",
        Colors.PURPLE
    )
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(
        f"Tavily Map: Successfully mapped {len(site_map['results'])} URLs from documentation site"
    )
    url_batches = chunk_urls(list(site_map['results']), chunk_size=20)
    log_info(
        f"URL processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches",
        Colors.BLUE
    )
    all_docs = await async_extract(url_batches)

if __name__=="__main__":
    asyncio.run(main())