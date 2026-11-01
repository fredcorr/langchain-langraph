from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from tavily import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3:latest"

load_dotenv()


@traceable(run_type="tool", name="Get product price")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"Looking up price for product: {product}")
    price_catalog = {
        "laptop": 999.99,
        "smartphone": 699.99,
        "headphones": 199.99,
        "monitor": 299.99,
    }
    return price_catalog.get(product.lower(), 0.0)


@traceable(run_type="tool", name="Apply discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the discounted price. Available discount tiers are bronze, silver, and gold."""
    print(f"Applying discount tier: {discount_tier} to price: {price}")
    if price <= 0:
        return 0.0
    discount_tiers = {
        "bronze": 5,
        "silver": 12,
        "gold": 23,
    }

    discount_percentage = discount_tiers.get(discount_tier.lower(), 0)
    discount_amount = price * (discount_percentage / 100)
    return max(price - discount_amount, 0.0)


@traceable(name="Product Price and Discount Agent")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model=f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=[
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ]
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\nIteration {iteration} of {MAX_ITERATIONS}")
        ai_messages = llm_with_tools.invoke(messages)
        tool_calls = ai_messages.tool_calls

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_messages.content}")
            return ai_messages.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        print(f"  [Tool Call ID] {tool_call_id}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        messages.append(ai_messages)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Welcome to the Product Price and Discount Agent!")
    run_agent("What is the price of a laptop and apply a gold discount?")
