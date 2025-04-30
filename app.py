# app.py

# ----------------------------------------------
# 1. Setup
import os
import uuid
import streamlit as st
from dotenv                                  import load_dotenv
from typing                                  import Annotated, Literal
from typing_extensions                       import TypedDict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools                    import tool
from langchain_experimental.utilities        import PythonREPL
from langchain_openai                        import ChatOpenAI
from langgraph.graph                         import MessagesState, StateGraph, START, END
from langgraph.types                         import Command
from langgraph.prebuilt                      import create_react_agent
from langchain_core.messages                 import HumanMessage,AIMessage
from langgraph.checkpoint.memory             import InMemorySaver 

# ----------------------------------------------
# 2. Set API Keys
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# ----------------------------------------------
# 3. Create Tools
tavily_tool = TavilySearchResults(api_key=tavily_api_key, max_results=5)
repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "The python code to execute to generate your chart."],):
    """Use this to execute python code and do math. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str

# ----------------------------------------------
# 4. Create Supervisor (Dynamic)

class Router(TypedDict):
    next: Literal["planner", "executor", "ideator", "writer", "FINISH"]

llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4.1-nano")

class State(MessagesState):
    next: str
    mode: str
    thread_id: str
    steps: int

def supervisor_node(state: State) -> Command[Literal["planner", "executor", "ideator", "writer", "__end__"]]:
    if "steps" not in state:
        state["steps"] = 0

    if state["steps"] >= 3:
        return Command(goto=END)

    allowed = {
        "solucionador": ["planner", "executor"],
        "criativo": ["ideator", "writer"]
    }[state["mode"]]

    system_prompt = (
        f"You are a supervisor managing workers in mode '{state['mode']}'.\n"
        f"Only alternate between: {', '.join(allowed)}.\n"
        "Return only one of the valid workers per step. After both have acted once, return 'FINISH'."
    )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]


    if goto not in allowed + ["FINISH"]:
        raise ValueError(f"Agente inválido retornado pelo LLM: {goto} no modo {state['mode']}")

    return Command(goto=goto, update={"next": goto, "steps": state["steps"] + 1})

# ----------------------------------------------
# 5. Create Agents

checkpointer = InMemorySaver()

# Planner and Executor (Modo Solucionador)
planner_agent = create_react_agent(llm, tools=[tavily_tool], checkpointer=checkpointer, prompt="You are a planner. Create structured plans, but DO NOT execute or code.")
executor_agent = create_react_agent(llm, tools=[python_repl_tool], checkpointer=checkpointer, prompt="You are an executor. Execute plans or code. DO NOT plan.")

# Ideator and Writer (Modo Criativo)
ideator_agent = create_react_agent(llm, tools=[], checkpointer=checkpointer, prompt="You are an ideator. Brainstorm ideas, themes, or concepts. DO NOT write final texts.")
writer_agent = create_react_agent(llm, tools=[], checkpointer=checkpointer, prompt="You are a writer. Expand and transform ideas into full texts. DO NOT brainstorm.")


def planner_node(state: State) -> State:
    input_msgs = state["messages"]
    prompt = [{"role": "system", "content": "Você é um planejador virtual que ajuda a organizar ideias."}] + input_msgs
    response = llm.invoke(prompt)
    ai_msg = AIMessage(content=response.content, name="planner")
    return {**state,"messages": input_msgs + [ai_msg],}

def executor_node(state: State) -> State:
    input_msgs = state["messages"]
    prompt = [{"role": "system", "content": "Você é um executor que transforma planos em ações concretas."}] + input_msgs
    response = llm.invoke(prompt)
    ai_msg = AIMessage(content=response.content, name="executor")
    return {**state,"messages": input_msgs + [ai_msg],}

def ideator_node(state: State) -> State:
    input_msgs = state["messages"]
    prompt = [{"role": "system", "content": "Você é um ideador criativo que propõe ideias novas e inovadoras."}] + input_msgs
    response = llm.invoke(prompt)
    ai_msg = AIMessage(content=response.content, name="ideator")
    return {**state,"messages": input_msgs + [ai_msg],}

def writer_node(state: State) -> State:
    input_msgs = state["messages"]
    prompt = [{"role": "system", "content": "Você é um redator que organiza ideias em textos bem estruturados."}] + input_msgs
    response = llm.invoke(prompt)
    ai_msg = AIMessage(content=response.content, name="writer")
    return {**state,"messages": input_msgs + [ai_msg],}


# ----------------------------------------------
# 6. Build Graph

builder = StateGraph(State)
builder.add_node("supervisor", supervisor_node)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("ideator", ideator_node)
builder.add_node("writer", writer_node)

builder.add_edge(START, "supervisor")

graph = builder.compile(checkpointer=checkpointer)

# ----------------------------------------------
# 7. Streamlit App (Frontend)

# Configuração da página (layout mais amplo)
st.set_page_config(page_title="Agente Multitarefa", layout="wide")

# Título do modelo
st.title("🧠 Multi-Agent: Criativo ou Solucionador")

# Movendo o conteúdo da col1 para a sidebar
with st.sidebar:
    st.markdown("""
    ### Este sistema utiliza múltiplos agentes de IA para fornecer respostas baseadas em dois modos distintos: **Solucionador** e **Criativo**.<br><br>
    Escolha abaixo o modo que melhor se adapta à sua necessidade e interaja com o agente para obter respostas dinâmicas e personalizadas:
    - No modo **Solucionador**, o agente foca em fornecer respostas técnicas e precisas para problemas complexos, ajudando a resolver dúvidas objetivas.
    - No modo **Criativo**, o agente utiliza sua capacidade de geração de ideias inovadoras para explorar conceitos fora da caixa, ideal para brainstorming e geração de novas ideias.
    """, unsafe_allow_html=True)
    
    mode = st.radio("Escolha o modo:", ["solucionador", "criativo"])
    st.session_state.mode = mode

# Coluna principal - Seletor de modo e interação com o modelo
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mode" not in st.session_state:
    st.session_state.mode = "solucionador"  # Modo padrão

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # ID único da sessão

# Campo de entrada do usuário
user_input = st.chat_input("Digite sua pergunta:")

if user_input:
    # Recupera o histórico anterior e adiciona a nova mensagem
    previous_messages = st.session_state.messages
    messages = previous_messages + [HumanMessage(content=user_input)]

    # Configura o thread_id único para o checkpointer
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Executa o grafo multiagente
    result = graph.invoke({
        "messages": messages,
        "mode": st.session_state.mode,
        "steps": 0
    }, config)

    # Atualiza o histórico completo de mensagens (inclui as geradas pelos agentes)
    st.session_state.messages = result["messages"]
    
    # Captura a última resposta gerada (a mais recente)
    full_response = result["messages"][-1].content

    # Salva no histórico de exibição para interface (chat)
    st.session_state.chat_history.append((user_input, full_response))

    # Renderiza a conversa
    for user_msg, agent_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(agent_msg)