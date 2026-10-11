from dotenv import load_dotenv

from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()
# Load environment variables from .env file

tavily = TavilySearch()

class Source(BaseModel):
    """Schema for source used by the agent"""
    url: str = Field(description="The URL of the source")
    name: str = Field(description="The name of the source")

class AgentResponse(BaseModel):
    """Schema for the agent's response"""
    answer: str = Field(description="The answer provided by the agent")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent")


llm = ChatOpenAI(model="gpt-5.2", temperature=0)
tools = [tavily]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details?"
            ),
        },
        config={"recursion_limit": 5}
    )
    
    print(result)


if __name__ == "__main__":
    main()
