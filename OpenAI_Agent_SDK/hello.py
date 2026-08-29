import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_tracing_disabled
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# Load environment variables from the .env file
load_dotenv()
set_tracing_disabled(True)

# Retrieve the OpenRouter API key from environment variables
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize the AsyncOpenAI client configured for OpenRouter
client = AsyncOpenAI(
    api_key=openrouter_api_key,
    base_url="https://openrouter.ai/api/v1"
)

# Configure the model using a standard OpenRouter model
model = OpenAIChatCompletionsModel(
    model="openai/gpt-4o-mini", 
    openai_client=client
)

# 1. Define Tool 1: Weather Tool
@function_tool
def get_weather(city: str) -> str:
    """Get the current weather forecast for a given city."""
    city_normalized = city.strip().title()
    return f"34°C, Sunny and clear skies in {city_normalized}"

# 2. Define Tool 2: News Tool
@function_tool
def get_latest_news(topic: str) -> str:
    """Get the latest news headlines and updates about a specific topic."""
    return f"Latest headlines on '{topic}': Breakthrough developments reported globally in AI and tech."

# 3. Create the Main AI Agent with both tools
assistant_agent = Agent(
    name="Assistant Agent",
    instructions=(
        "You are a helpful personal assistant equipped with weather and news tools. "
        "Always use your tools when the user asks for real-time weather details or news updates."
    ),
    tools=[get_weather, get_latest_news],
    model=model
)

async def main():
    print("--- OpenRouter Agent Initialized ---")
    print("You can now chat with your agent directly from the terminal.")
    print("Type 'exit' or 'quit' to end the session.\n" + "-" * 50)
    
    while True:
        # Accept live input from the user in the terminal
        user_query = input("\nAsk your agent: ").strip()
        
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting agent session. Goodbye!")
            break
            
        if not user_query:
            print("Please enter a valid question.")
            continue

        print("\nProcessing your request...")
        
        try:
            # Run the agent with the user's input query
            result = await Runner.run(assistant_agent, user_query)
            print(f"\nAgent Response:\n{result.final_output}")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())