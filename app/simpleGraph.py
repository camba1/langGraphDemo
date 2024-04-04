from os import getenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
from utils.constants import OPENROUTER_API_BASE, OPENROUTER_MIXTRAL_MODEL_NAME, CREATIVE_MODEL_TEMP


openrouter_api_key = getenv("OPENROUTER_API_KEY")


def create_graph(model):
    graph = MessageGraph()

    graph.add_node("modelCall", model)
    graph.add_edge("modelCall", END)

    graph.set_entry_point("modelCall")
    return graph


def run_app():
    model = ChatOpenAI(temperature=CREATIVE_MODEL_TEMP,
                       model=OPENROUTER_MIXTRAL_MODEL_NAME,
                       openai_api_key=openrouter_api_key,
                       openai_api_base=OPENROUTER_API_BASE,
                       )
    graph = create_graph(model)
    compiled_graph = graph.compile()

    print(compiled_graph.invoke(HumanMessage("Who is Pinocchio")))


run_app()
