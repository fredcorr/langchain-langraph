from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()
# Load environment variables from .env file

tavily = TavilySearch()


llm = ChatOpenAI(model="gpt-5", temperature=0)
tools = [tavily]
agent = create_agent(model=llm, tools=tools)


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
