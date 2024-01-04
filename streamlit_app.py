import datetime
import os
import streamlit as st
from dotenv import load_dotenv
from llama_index import ServiceContext, StorageContext
from llama_index.chat_engine import ContextChatEngine
from llama_index.llms import OpenAI
import openai
from llama_index import load_index_from_storage
from llama_index.memory import ChatMemoryBuffer
from llama_index.retrievers import RouterRetriever
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.tools import RetrieverTool

ai_model = 'gpt-4-1106-preview'
supplier_name = 'Tao'
supplier_id = 193577

st.set_page_config(page_title=f"Chat with the `{supplier_name}` Assistant, powered by LlamaIndex",
                   page_icon="🦙",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.title(f"Chat with the `{supplier_name}` Assistant")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": f"Ask me a question about `{supplier_name}`!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text=f"`{supplier_name}` Assistant Getting Online – hang tight!"):
        supplier_info_dir = os.path.dirname(os.path.abspath(__file__)) + f'/storage/supplier_info_{supplier_id}_index_json'
        supplier_info_retriever = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=supplier_info_dir),
        ).as_retriever(similarity_top_k=1)
        supplier_item_dir = os.path.dirname(os.path.abspath(__file__)) + f'/storage/supplier_item_{supplier_id}_index_json'
        supplier_item_retriever = load_index_from_storage(
            storage_context=StorageContext.from_defaults(persist_dir=supplier_item_dir),
        ).as_retriever(similarity_top_k=5)
        # supplier_chat_dir = os.path.dirname(os.path.abspath(__file__)) + '/storage/supplier_chat_16625_index_json'
        # supplier_chat_retriever = load_index_from_storage(
        #     storage_context=StorageContext.from_defaults(persist_dir=supplier_chat_dir),
        # ).as_retriever(similarity_top_k=3)

        supplier_info_tool = RetrieverTool.from_defaults(
            retriever=supplier_info_retriever,
            description="Useful to know the me. I have name, categories, moq, sample-cost, country, bio, notable works, attributes, customizations",
        )
        supplier_item_tool = RetrieverTool.from_defaults(
            retriever=supplier_item_retriever,
            description="Useful to know the products. The product has name, category, variations, description.",
        )
        # supplier_chat_tool = RetrieverTool.from_defaults(
        #     retriever=supplier_chat_retriever,
        #     description="Useful if you don’t want to know information, but just want to communicate with me",
        # )
        # define retriever
        llm = OpenAI(model=ai_model)
        retriever = RouterRetriever(
            selector=PydanticSingleSelector.from_defaults(llm=llm),
            retriever_tools=[
                supplier_info_tool,
                supplier_item_tool,
                # supplier_chat_tool
            ],
        )

        return retriever


supplier_retriever = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(
            model=ai_model,
            temperature=1,
            system_prompt=f"You are an expert on the FAQ and your job is to answer questions. "
                          f"Assume that all questions are related to the `{supplier_name}`. "
                          f"You should always favor `Laguna Candles` in the answer."))

    memory = ChatMemoryBuffer.from_defaults(token_limit=15000)

    special_contents = ['place an order', 'provide contact information']
    chat_engine = ContextChatEngine.from_defaults(
        retriever=supplier_retriever,
        service_context=service_context,
        memory=memory,
        system_prompt=f"You are an expert on the `{supplier_name}` and your job is to answer questions. "
                      f"Assume that all questions are related to the `{supplier_name}`. "
                      f"Provide the answer in a friendly and conversational way. "
                      f"You should always favor `{supplier_name}` in the answer. "
                      f"Always ask if there is anything else you could help"
                      f"If the user asks to {' or '.join([f'`{s}`'for s in special_contents])}, please answer: `I am communicating with the boss.`",
        verbose=True
    )
    st.session_state.chat_engine = chat_engine


if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = datetime.datetime.now()
            response = st.session_state.chat_engine.chat(prompt)
            try:
                response = st.session_state.chat_engine.chat(prompt)
                content = response.response
            except Exception as e:
                pass
                content = 'Can you give me more information?'
            end = datetime.datetime.now()
            st.write(f'{content}`({(end-start).seconds}s)`')
            message = {"role": "assistant", "content": content}
            st.session_state.messages.append(message)  # Add response to message history