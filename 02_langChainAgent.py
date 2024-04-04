
# 필요한 라이브러리 및 모듈을 가져온다.
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

#1: Ollama LLM 모델 및 기타 관련 객체 초기화
llm = Ollama(model="llama2")
#모델의 출력을 분석하여 이해하기 쉬운 형식으로 변환하는 역할
output_parser = StrOutputParser()
# 웹에서 문서를 로드하기 위한 객체를 초기화하고 지정된 URL에서 문서를 가져온다.
loader = WebBaseLoader("https://docs.smith.langchain.com")
# 웹에서 문서를 로드하고, 그 결과를 docs 변수에 저장한다.
docs = loader.load()
# 모델에서 단어를 벡터 형식으로 변환하는데 사용
embeddings = OllamaEmbeddings()
# 클래스를 사용하여 문서를 재귀적으로 문자로 분할하는 객체를 초기화
text_splitter = RecursiveCharacterTextSplitter()

#2: 문서 분할 및 FAISS 벡터화를 위한 초기화
# 문자로 분할하고, 결과를 documents 변수에 저장
documents = text_splitter.split_documents(docs)
# 문서를 벡터로 변환한다. 이 때, embeddings 객체를 사용하여 문서를 벡터 형식으로 임베딩
vector = FAISS.from_documents(documents, embeddings)
# 검색 작업을 수행할 수 있는 검색 엔진(retriever)을 생성하고, 결과를 retriever 변수에 저장 문서간의 유사도를 계산
retriever = vector.as_retriever()

# 3: 초기 대화 내용 설정
# 이전의 대화 내용을 사전에 제공한다.
# 이를 통해 대화 추적이 가능하도록 하며 상황 이해를 쉽게 할 수 있어 이를 통해 자연스러운 상호작용이 가능하여 대화의 맥락 유지가 가능해진다.
chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
<context>
{context}
</context>
Question: {input}""")

# 4: 대화 기록을 포함한 검색 체인 설정
# 사용자에게 입력을 받을 때 사용되는 템플릿을 설정한다. 
# 주어진 context에 기반하여 특정 질문에 대한 응답을 생성하게된다.
prompt_with_history = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
# 이전의 대화 내용을 고려하여 검색 쿼리를 생성하고 대화와 관련된 정보를 제공하는데 도움을 준다.
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_with_history)

# 5: 문서 처리를 위한 체인 설정
prompt_for_document = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
# 문서를 처리하고 사용자의 질문에 문서를 기반으로 답변할 수 있는 체인을 설정한다.
document_chain = create_stuff_documents_chain(llm, prompt_for_document)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 위 코드는 이전의 내용을 정리하였으며 아래의 코드부 agent 생성 코드이다.
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,# 검색을 수행할 검색기 
    "langsmith_search", # 검색 도구의 이름을 지정하는 문자 
    "artificial intelligence expert",# 검색 도구의 대한 설명
)

# 대리자 생성을 위해 키 발급
import getpass
import os
os.environ["TAVILY_API_KEY"] = getpass.getpass() # Tavily 키 발급 받고 실행시 password 입력 부분에 입력하면됨
os.environ["OPENAI_API_KEY"] = "open ai 키 넣으면됨"
from langchain_community.tools.tavily_search import TavilySearchResults
# https://app.tavily.com/ 회원 가입후 키 발급
search = TavilySearchResults()

# 작업 도구 목록 생성
tools = [retriever_tool, search]

from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_functions_agent,create_react_agent
from langchain.agents import AgentExecutor
# 모델 선택
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 에이전트에 사용할 프롬프트 가져오기 
prompt_for_openai_agent = hub.pull("hwchase17/openai-functions-agent")
# 에이전트 생성
agent_apenai = create_openai_functions_agent(llm, tools, prompt_for_openai_agent)
# 에이전트 실행을 위한 executor 생성
agent_executor = AgentExecutor(agent=agent_apenai, tools=tools, verbose=True)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
response=agent_executor.invoke({
    "chat_history": chat_history,
    "input": "Tell me how"
})
print(response)



# 다른 에이전트에 사용할 프롬프트 가져오기
prompt_for_retrieval_agent = hub.pull("hwchase17/react-chat")
# 다른 에이전트 생성
retrieval_agent = create_react_agent(llm, tools, prompt_for_retrieval_agent)
# 다른 에이전트 excutor 설정
retrieval_agent_executor = AgentExecutor(agent=retrieval_agent, tools=tools, verbose=True)
response=agent_executor.invoke({
    "chat_history": chat_history,
    "input": "how is the weather today?"
})
print(response)


