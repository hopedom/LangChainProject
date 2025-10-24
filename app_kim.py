import os
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


os.environ["OPENAI_API_KEY"] = ""


@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-5-mini")


@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')
    persist_directory = 'chroma_db_pd_sklearn'
    return Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)


@st.cache_resource
def get_rag_chain():
    vectorstore = get_vector_store()
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    
    RAG_PROMPT_TEMPLATE = """
    당신은 Pandas 2.3, Scikit-Learn 1.7.2의 권위자입니다. 오로지 Pandas, Scikit-Learn에 대해서만 답변하세요.
    Pandas, Scikit-Learn과 관련이 없는 질문에 대해서는 "해당 내용은 공식 문서에 없습니다."라고 답변하세요.
    반드시 아래에 제공되는 [검색된 Pandas, Scikit-Learn 문서 내용]을 근거로 하여 사용자의 질문에 답변해야 합니다.

    [검색된 Pandas, Scikit-Learn 문서 내용]:
    {context}

    [사용자 질문]:
    {question}

    [답변]:
    """
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    def format_docs(docs):
        return "\n\n".join(
            f"--- (출처: {doc.metadata.get('source', 'N/A')}) ---\n" + doc.page_content
            for doc in docs
        )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


st.set_page_config(page_title="Pandas 2.3 & Scikit-Learn 1.7.2 RAG 챗봇")
st.title("Pandas & Scikit-Learn\n공식 문서 기반 어시스턴트")
st.caption("Pandas, Scikit-Learn에 관련된 것은 무엇이든 물어보세요!")

try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"RAG Chain 로드 중 오류 발생: {e}")
    st.stop()

# 1. 채팅 기록 초기화 (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! Pandas & Scikit-Learn 공식 문서에 기반하여 질문에 답변해 드립니다."}
    ]

# 2. 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력 받기 (chat_input)
if prompt := st.chat_input("Pandas, Scikit-Learn에 대해 질문하세요..."):
    # 3-1. 사용자 메시지를 채팅 기록에 추가하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3-2. RAG Chain을 실행하여 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("Pandas, Scikit-Learn 문서를 검색하고 답변을 생성 중입니다..."):
            try:
                # [핵심] RAG Chain 실행!
                response = rag_chain.invoke(prompt)
                
                # (팁) 답변과 함께 근거 문서를 보여주면 신뢰도가 올라갑니다.
                # response_with_source = f"{response}\n\n**[답변 근거]**\n(근거 문서는 `rag_chain`을 수정하여 `context`를 함께 반환하도록 해야 합니다.)"
                
                st.markdown(response)
                # 채팅 기록에 AI 답변 추가
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"답변 생성 중 오류 발생: {e}")