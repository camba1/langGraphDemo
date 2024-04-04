from typing import TypedDict, Union, Annotated
import operator
import logging
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.messages import BaseMessage
from langchain_core.runnables.base import Runnable
from langchain_core.agents import AgentAction, AgentFinish
from langgraph.graph import END, StateGraph
from utils.constants import FACTUAL_MODEL_TEMP, OPENAI_GPT35_MODEL_NAME, HUB_TOOLS_PROMPT, LOG_FORMAT
from langchain.agents import create_openai_functions_agent
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph.graph import CompiledGraph

logging.basicConfig(level=logging.WARNING, format=LOG_FORMAT)

model = ChatOpenAI(temperature=FACTUAL_MODEL_TEMP, model_name=OPENAI_GPT35_MODEL_NAME)


class AgentState(TypedDict):
    # The input string
    input: str
    # The list of previous messages in the conversation
    chat_history: list[BaseMessage]
    # The outcome of a given call to the agent
    # Needs `None` as a valid type, since this is what this will start as
    agent_outcome: Union[AgentAction, AgentFinish, None]
    # List of actions and corresponding observations
    # Here we annotate this with `operator.add` to indicate that operations to
    # this state should be ADDED to the existing values (not overwrite it)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


# Define the tools we are going to use the decorator makes it easier to declare them
@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int


@tool
def add(first_int: int, second_int: int) -> int:
    """Add two integers."""
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    """Update the base to the exponent power."""
    return base ** exponent


def build_agent() -> (Runnable, ToolExecutor):
    """
    Build the agent executor that will be responsible to answer the user"s question
    :return: The agent executor object
    """

    prompt = hub.pull(HUB_TOOLS_PROMPT)
    # prompt.pretty_print()

    tools = [multiply, add, exponentiate]

    # Construct the Tools agent
    # agent = create_openai_tools_agent(model, tools, prompt)
    agent = create_openai_functions_agent(model, tools, prompt)
    logging.info("Agent created")

    tool_executor = ToolExecutor(tools)

    logging.info("Tool Executor created")

    return agent, tool_executor


# Define the agent
def run_agent(data):
    agent_outcome = agent.invoke(data)
    return {"agent_outcome": agent_outcome}


# Define the function to execute tools
def execute_tools(data):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = data["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}


# Define logic that will be used to determine which conditional edge to go down
def should_continue(data):
    print(f"data: {data} \n")
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data["agent_outcome"], AgentFinish):
        return "end"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "continue"


def create_graph() -> CompiledGraph:
    logging.info("Creating graph")
    graph = StateGraph(AgentState)

    logging.info("Adding nodes")
    # Define the two nodes we will cycle between
    graph.add_node("agent", run_agent)
    graph.add_node("action", execute_tools)

    graph.set_entry_point("agent")

    logging.info("Adding edges")
    graph.add_edge("action", "agent")

    # We now add a conditional edge
    graph.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )
    logging.info("Compiling graph")
    compiled_graph = graph.compile()
    logging.info("Compiled graph")
    return compiled_graph


def app():
    graph = create_graph()
    logging.info("Running graph")
    question = {
        "input": "Take 3 to the fifth power and multiply that by the sum of twelve and three,"
                 " then square the whole result",
        "chat_history": []
    }
    # result = graph.invoke(question)
    # print(result)
    for s in graph.stream(question):
        print(list(s.values())[0])
        print("----")


agent, tool_executor = build_agent()
app()

