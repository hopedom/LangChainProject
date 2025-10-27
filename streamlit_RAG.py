import os
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# RAG_system.py에서 지정한 DB 경로
PERSIST_DIRECTORY = "./chroma_db_multi_source"

METADATA_KEYS = {
    "pandas": "Pandas",
    "scikit_learn": "Scikit Learn",
    "pytorch": "Pytorch",
    "python": "Python"
}


@st.cache_resource
def get_vectorstore():
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"Vector DB 디렉토리를 찾을 수 없습니다: {PERSIST_DIRECTORY}")
        st.info("먼저 RAG_system.py 스크립트를 실행하여 벡터 DB를 생성해야 합니다.")
        return None
    
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-small')
    
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )

@st.cache_resource
def get_base_components():
    qa_system_prompt = """
    당신은 '컨텍스트'로 제공된 정보만을 기반으로 답변하는 매우 엄격한 AI 어시스턴트입니다.
    당신의 사전 지식이나 외부 인터넷 정보는 그 어떤 경우에도 절대 사용해서는 안 됩니다.

    ---
    [컨텍스트]:
    {context}
    ---

    [규칙]:
    1. 질문에 대한 답변은 반드시, 오직, 100% 위 [컨텍스트]에서만 가져와야 합니다.
    2. [컨텍스트]에 질문에 대한 답변이 명시적으로 존재하지 않는 경우, 
       절대로 외부 지식을 활용해 답변하지 말고, "죄송합니다. 제공된 문서 내에서 해당 정보를 찾을 수 없습니다."라고만 답변해야 합니다.
    3. [컨텍스트]의 내용을 기반으로 추론하거나 요약할 수는 있지만, [컨텍스트]에 없는 새로운 정보(예: 관련 코드 예시, 추가 설명)를 절대 생성하거나 추가해서는 안 됩니다.
    4. 모든 답변은 한국어로 정중하게 작성해야 합니다.

    이제 위 [컨텍스트]와 [규칙]을 철저히 준수하여 다음 질문에 답변하세요.
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [("system", qa_system_prompt), ("human", "{input}")]
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    return llm, qa_prompt

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



st.header("다중 소스 RAG 챗봇")
st.caption("검색 범위를 선택하여 질문하세요.")


vectorstore = get_vectorstore()
llm, qa_prompt = get_base_components()

st.sidebar.title("검색 범위 설정")
selected_options = {}
for key, label in METADATA_KEYS.items():
    # UI에 "Pandas", "Scikit Learn" 등 표시
    selected_options[key] = st.sidebar.checkbox(label, value=True)


selected_filters = []
for key, is_selected in selected_options.items():
    if is_selected:
        # 필터링은 "pandas", "scikit_learn" 등 내부 키로 수행
        selected_filters.append({"source_type": {"$eq": key}})

if not selected_filters:
    filter_dict = {"source_type": {"$eq": "INVALID_TAG_NO_SELECTION"}}
    st.sidebar.warning("하나 이상의 검색 범위를 선택하세요.")
elif len(selected_filters) == 1:
    filter_dict = selected_filters[0]
else:
    filter_dict = {"$or": selected_filters}

if vectorstore:
    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": filter_dict}
    )

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
else:
    rag_chain = None
    st.error("챗봇을 초기화할 수 없습니다. RAG_system.py를 실행하세요.")

# 5. 챗봇 메시지 기록
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "어떤 소스에서 정보를 찾아드릴까요?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# 6. 챗봇 입력 처리
if prompt_message := st.chat_input("질문을 입력해주세요..."):
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    
    if rag_chain and selected_filters:
        with st.chat_message("ai"):
            with st.spinner("선택하신 소스에서 검색 중..."):
                response = rag_chain.invoke(prompt_message)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
    elif not selected_filters:
         st.error("검색할 소스를 사이드바에서 1개 이상 선택해주세요.")
    else:
        st.error("챗봇이 준비되지 않았습니다. 관리자에게 문의하세요.")