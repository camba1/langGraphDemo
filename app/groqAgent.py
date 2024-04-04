import json
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.tools import tool
from utils.constants import GROQ_MIXTRAL_MODEL_NAME, LOG_FORMAT, FACTUAL_MODEL_TEMP

logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

# We assume there is a groq API Key in the environment already setup
MODEL_NAME = GROQ_MIXTRAL_MODEL_NAME


# Example dummy function (tool) hard coded to return the score of an NBA game. The scores are not real, of course,
# but make it obvious that the tool is working
@tool
def get_game_score(team_name: str) -> str:
    """Get the score for a given NBA game."""
    logging.info(f"get_game_score accessed: team name: {team_name}")
    if "warriors" in team_name.lower():
        return json.dumps(
            {"game_id": "401585601", "status": 'Final', "home_team": "Los Angeles Lakers", "home_team_score": 1230,
             "away_team": "Golden State Warriors", "away_team_score": 1234})
    elif "lakers" in team_name.lower():
        return json.dumps(
            {"game_id": "401585601", "status": 'Final', "home_team": "Los Angeles Lakers", "home_team_score": 1400,
             "away_team": "Golden State Warriors", "away_team_score": 2300})
    elif "nuggets" in team_name.lower():
        return json.dumps({"game_id": "401585577", "status": 'Final', "home_team": "Miami Heat", "home_team_score": 880,
                           "away_team": "Denver Nuggets", "away_team_score": 1000})
    elif "heat" in team_name.lower():
        return json.dumps({"game_id": "401585577", "status": 'Final', "home_team": "Miami Heat", "home_team_score": 880,
                           "away_team": "Denver Nuggets", "away_team_score": 1000})
    else:
        return json.dumps({"team_name": team_name, "score": "unknown"})


# Define the prompt that will be sent to the LLM. This includes a scratch pad for the agent
# to store any intermediate steps while fulfilling the request.
def create_prompt() -> ChatPromptTemplate:

    system_msg = ("You are a function calling LLM that uses the data extracted from the "
                  "get_game_score function to answer questions around NBA game scores."
                  "Include the team and their opponent in your response."
                  )
    human_msg = "{user_prompt}"
    agent_scratchpad = MessagesPlaceholder(variable_name="agent_scratchpad")
    prompt = ChatPromptTemplate.from_messages([system_msg, human_msg, agent_scratchpad])

    return prompt


# Call the LLM and have it tell us that we need to call the tool.
# Then, run the tool and finally return the response
def run_conversation(user_prompt):

    # Step 1: Create the prompt that will be sent to the LLM.
    preprocessor = {
        "user_prompt": lambda x: x["user_prompt"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"] if "intermediate_steps" in x else ""
        ),
    }
    prompt = create_prompt()
    logging.info("Prompt: %s", prompt.invoke(input={"user_prompt": user_prompt, "agent_scratchpad": []}))

    # Step 2: Setup model and associate it with the tool
    tools = [get_game_score]
    model = ChatGroq(temperature=FACTUAL_MODEL_TEMP, model_name=MODEL_NAME)
    model_with_tools = model.bind_tools(tools)

    # Step 3: Define the agent
    chain = preprocessor | prompt | model_with_tools | OpenAIToolsAgentOutputParser()
    response_message = AgentExecutor(agent=chain, tools=tools).invoke({"user_prompt": user_prompt})
    return response_message


def run_app():
    user_prompt = "What was the score of the warriors game?"
    print(run_conversation(user_prompt))


run_app()
