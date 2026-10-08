from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from .env file

def main():
    print("Hello from langchain-course!")
    information = "Musk was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated in January 2025, Musk was Senior Advisor to the President and the de facto head of the Department of Government Efficiency (DOGE). Musk left the Trump administration in May 2025 and returned to managing his companies; shortly thereafter he had a public feud with Trump."
    summary_template = """given the information: {information}, about a person I want you to create: \
        1. A short summary
        2. Two interesting facts
    """

    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    # llm = ChatOllama(model="gemma3:270m", temperature=0)
    llm = ChatOpenAI(model="gpt-5", temperature=0)
    chain = prompt | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()