from operator import itemgetter
import os

from langchain.chat_models import ChatOllama
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document

import json
from typing import Iterable

import streamlit as st

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RAG with Private sLLM",
    page_icon="🤖",
)

st.title("규정 검색 (Model: Private sLLM)")

def get_model_name(model_choice):
    if model_choice == "kor-univ-kullm3-10.7b":
        model_name = "kullm3:latest"
    elif model_choice == "yanolja-eeve-10.8b":
        model_name = "eeve:latest"
    else: model_name = None
    return model_name

with st.sidebar:
    model_choice = st.selectbox(
            label="사용할 sLLM 모델을 선택하세요.",
            options=["kor-univ-kullm3-10.7b", "yanolja-eeve-10.8b"],
            index=None,
            placeholder="모델을 선택하세요...",
        )
    
    st.session_state["model"] = get_model_name(model_choice)

if st.session_state["model"] is None:
    st.caption("＊좌측 상단의 사이드바에서 모델을 선택하세요.")
    st.stop()
else:
    st.caption("＊채팅 기록을 삭제하려면 새로고침을 해주세요.")

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOllama(
    model=st.session_state["model"],
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

memory_llm = ChatOllama(
    model=st.session_state["model"],
    temperature=0.1,
)

if "private_messages" not in st.session_state:
    st.session_state["private_messages"] = []

if "private_memory" not in st.session_state:
    st.session_state["private_memory"] = ConversationSummaryBufferMemory(
        llm=memory_llm,
        max_token_limit=200,
        memory_key="chat_history",
        return_messages=True,
    )

@st.cache_resource(show_spinner="규정을 불러오고 있습니다...")
def embed_file():
    file_name = "instruction.pdf"
    file_path = f"./files/{file_name}"
    docs_path = f"./.cache/{model_choice}_embeddings/{file_name}/data.jsonl"
    
    cache_dir = LocalFileStore(f"./.cache/{model_choice}_embeddings/{file_name}")

    # splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     separator="\n",
    #     chunk_size=600,
    #     chunk_overlap=100,
    # )

    # loader = UnstructuredFileLoader(file_path)
    # docs = loader.load_and_split(text_splitter=splitter)

    # def save_docs_to_jsonl(array:Iterable[Document], docs_path:str)->None:
    #     os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    #     with open(docs_path, 'w') as jsonl_file:
    #         for doc in array:
    #             jsonl_file.write(doc.json() + '\n')

    # save_docs_to_jsonl(docs, docs_path)
    
    def load_docs_from_jsonl(docs_path)->Iterable[Document]:
        array = []
        with open(docs_path, 'r') as jsonl_file:
            for line in jsonl_file:
                data = json.loads(line)
                obj = Document(**data)
                array.append(obj)
        return array

    docs = load_docs_from_jsonl(docs_path)

    embeddings = OllamaEmbeddings(
        model=st.session_state["model"]
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()

    #STUFF IS TOO LONG FOR SLLMS

    return retriever

def save_message(message, role):
    st.session_state["private_messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["private_messages"].append({"message":message, "role":role})

def paint_history():
    for message in st.session_state["private_messages"]:
        send_message(message["message"], message["role"], save=False,)

# def format_docs(docs):
#     return "\n\n".join(document.page_content for document in docs)

retriever = embed_file()

send_message("저는 「국방정보화업무 훈령」 검색 챗봇입니다. 무엇이든 물어보세요!", "ai", save=False)
paint_history()


message = st.chat_input("질문을 입력하세요...")

map_doc_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use the following portion of a long document to see if any of the text is relevant to answer the question. Return any relevant text verbatim. If there is no relevant text, return : ''
            -------
            {context}
            """
        ),
        ("human", "{question}"),
    ]
)

map_doc_chain = map_doc_prompt | llm

# Apply MapReduce to the retrieved documents
def map_docs(inputs):
    documents = inputs["documents"]
    question = inputs["question"]
    return "\n\n".joint(
        map_doc_chain.invoke(
            {"context": doc.page_content, "question": question}
        ).content
        for doc in documents
    )

map_chain = {
    "documents": retriever,
    "question": RunnablePassthrough(),
} | RunnableLambda(map_docs)

prompt = ChatPromptTemplate.from_messages([
    ("system", 
    """
    Answer the question using ONLY the provided context. 
    Answer in Korean ONLY.
    If you don't know the answer, just say you don't know. DON'T make anything up.

    Context: {context}
    -----
    And you will get a summarized context of the chat history. If it's empty, you don't have to care.
    
    Chat history: {chat_history}
    """
    ),
    ("human", "{question}")
])

try:
  if message:
    send_message(message, "human")

    chain = {
        "context": map_chain,
        "question": RunnablePassthrough(),
    } | RunnablePassthrough.assign(
        chat_history=RunnableLambda(
            st.session_state["private_memory"].load_memory_variables
            ) | itemgetter("chat_history")
        ) | prompt | llm

    with st.chat_message("ai"):
        response = chain.invoke(message)
        st.session_state["private_memory"].save_context(
            {"input": message}, 
            {"output": response.content},
        )
except:
  # Prevent the error from propagating into your Streamlit app.
  pass