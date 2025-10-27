import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# OpenAI API 키 읽기 (환경변수 또는 Streamlit secrets)
api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.error("OPENAI_API_KEY가 설정되어 있지 않습니다. 환경변수 또는 Streamlit secrets에 추가하세요.")
os.environ["OPENAI_API_KEY"] = api_key

# Chroma DB 경로들 (jin_data.py에서 생성한 경로와 일치)
PANDAS_CHROMA_DIR = "./chromadb/pandas_rst"
SKLEARN_CHROMA_DIR = "./chromadb/sklearn_rst"
BOTH_CHROMA_DIR = "./chromadb/both_rst"
EMB_MODEL = "text-embedding-3-small"

@st.cache_resource
def load_vectorstore_for_choice(choice: str):
    """choice: 'both' | 'pandas' | 'sklearn'"""
    path_map = {
        "both": BOTH_CHROMA_DIR,
        "pandas": PANDAS_CHROMA_DIR,
        "sklearn": SKLEARN_CHROMA_DIR,
    }
    persist_dir = path_map.get(choice)
    if not persist_dir or not os.path.exists(persist_dir):
        st.error(f"선택한 데이터베이스가 없습니다: {persist_dir}")
        return None
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

@st.cache_resource
def initialize_components(selected_model: str, data_choice: str):
    llm = ChatOpenAI(model=selected_model, temperature=0, openai_api_key=api_key)
    vectorstore = load_vectorstore_for_choice(data_choice)
    if vectorstore is None:
        return None

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 히스토리 문맥화용 프롬프트 (질문 재정의)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # QA 시스템 프롬프트: 오직 제공된 Context만 사용하도록 강제
    qa_system_prompt = """다음에 제공된 'Context' 안의 내용만을 근거로 질문에 답변하세요.
만약 Context에 답이 없으면 '모르겠습니다.'라고 답하세요.
추가적인 정보를 추측하거나 문서 외부의 내용을 생성하지 마세요.
응답은 한국어 존댓말로 작성하세요.

{context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

def test_vectorstore_search_for(choice: str):
    vs = load_vectorstore_for_choice(choice)
    if vs is None:
        return
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    queries = {
        "pandas": "Pandas에서 누락된 값(Missing Values)을 확인하는 가장 일반적인 메서드는 무엇인가요?",
        "sklearn": "scikit-learn에서 estimator 사용법의 기본은 무엇인가요?",
        "both": "DataFrame과 estimator를 함께 사용하는 방법"
    }
    q = queries.get(choice, "예시 질의")
    docs = retriever.get_relevant_documents(q)
    st.write("검색 결과 문서 개수:", len(docs))
    for i, d in enumerate(docs):
        st.write(f"- {i+1}. {d.metadata.get('source','알 수 없음')} [{d.metadata.get('library','')}]")
        st.code(d.page_content[:400].strip() + ("..." if len(d.page_content) > 400 else ""))

# Streamlit UI
st.header("Python 개발 라이브러리 공식문서 LLM 검색기")
st.markdown("대화형 질의를 통해 Pandas, Scikit-learn 문서를 검색하고 답변을 얻을 수 있습니다.")

data_choice_label = st.radio(
    "어떤 데이터를 사용할까요?",
    ("1) Pandas+Scikit-learn", "2) Pandas", "3) Scikit-learn")
)
# map radio selection to internal keys
choice_map = {
    "1) Pandas+Scikit-learn": "both",
    "2) Pandas": "pandas",
    "3) Scikit-learn": "sklearn",
}
data_choice = choice_map[data_choice_label]

option = st.selectbox("Select GPT Model", ("gpt-4o-mini", "gpt-3.5-turbo-0125"))
selected_model = option


# 디버깅 출력: 실제로 어떤 폴더를 로드하는지 확인
path_map = {"both": BOTH_CHROMA_DIR, "pandas": PANDAS_CHROMA_DIR, "sklearn": SKLEARN_CHROMA_DIR}
persist_dir = path_map.get(data_choice)
st.write("선택된 데이터:", data_choice, "→", persist_dir)

if not persist_dir or not os.path.exists(persist_dir):
    st.warning("선택한 벡터스토어 디렉토리가 없습니다. 데이터 생성 스크립트가 정상 실행되었는지 확인하세요.")
else:
    try:
        st.write("디렉토리 파일/항목 수:", len(os.listdir(persist_dir)))
    except Exception as e:
        st.write("디렉토리 확인 중 오류:", e)


if st.button("벡터스토어 로드 확인"):
    test_vectorstore_search_for(data_choice)

rag_chain = initialize_components(option, data_choice)
if rag_chain is None:
    st.stop()

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "공식문서 기반으로 질문에 답해드립니다. 무엇을 도와드릴까요?"}]

# 기존 히스토리 출력 (Streamlit chat UI)
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt_message := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(prompt_message)
    with st.chat_message("ai"):
        with st.spinner("검색 중..."):
            config = {"configurable": {"session_id": "any"}}
            response = conversational_rag_chain.invoke({"input": prompt_message}, config)
            answer = response.get('answer') or response.get('output') or ""
            st.write(answer)
            with st.expander("참고 문서 확인"):
                docs = response.get('context') or response.get('source_documents') or []
                for doc in docs:
                    src = doc.metadata.get('source', '알 수 없음')
                    lib = doc.metadata.get('library', '')
                    st.markdown(f"- {src} [{lib}]")