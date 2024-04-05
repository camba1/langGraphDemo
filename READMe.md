# LangGraph demo

This repository shows a number of AI powered LangChain agent and LangGraph graph examples. 
There are two other sister repositories that show additional LangChain functionality:
- [LangChainDemo](https://github.com/camba1/langChainDemo) shows a number of examples of using LangServe, LangChain and LangSmith
- [LangChainDemoClient](https://github.com/camba1/langchainDemoClient) shows a few simple frontend applications (Gradio, Streamlit, ChainLit) to connect 
to the LangServe application


## Quick start

Make sure you have an API for [OpenAI](https://openai.com/product) and one for [Groq](https://wow.groq.com) as there are 
examples using one or the other platform inference points.

Assuming that you have [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) installed, you can set up the project as shown below:

```shell
git clone https://github.com/camba1/langGraphDemo.git
cd langGraphDemo
poetry install
poetry shell
export OPENAI_API_KEY=<YOUR OPENAI_API_KEY>
export GROQ_API_KEY=<YOUR GROQ_API_KEY>    
```

If you want to trace how your application is working, you can set up LangSmith setting up the three 
environment variables shown below:

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<YOUR LANGCHAIN_API_KEY>
export LANGCHAIN_PROJECT=<A_NAME_FOR_YOUR_PROJECT_CAN_BE_ANYTHING> 
```

Then you can run the different examples as follows:

- **calculate**: Interact with OpenAI and run a graph that does simple math. Run with: `python app/calculate.py` 
- **groqAgent**: Uses Groq to call tools using Mixtral-8x7B. Run using: `python app/groqAgent.py`
- **SimpleGraph**: A very simple graph that runs Using OpenAI. Run with: `python app/simpleGraph.py` 