from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily=TavilyClient()

@tool
def search(query: str) -> str:
    """Search for real-time information."""
    print(f"Searching for {query}")
    return tavily.search(query=query)
    

llm = ChatOpenAI(model="gpt-5")
tools = [search]

agent = create_react_agent(llm, tools)

def main1():
    print("Hello from Langchain")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for 3 job postings for AI Engineer using Langchain near Bengaluru, KA on LinkedIn and list the details for me")]
    })
    print(result["messages"][-1].content)
    #print(result)

if __name__ == "__main__":
    main1()