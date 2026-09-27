from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LLM
llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

# MEMÓRIA
history = InMemoryChatMessageHistory()

# PDF
loader = PyPDFLoader(
    "data/Curriculo_Caua_Dantas_Barbosa.pdf"
)

docs = loader.load()

# SPLITTER
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# EMBEDDINGS
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# CHROMADB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# RETRIEVER
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# PROMPT
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Você é um especialista em responder perguntas
        sobre o conteúdo dos documentos fornecidos.

        Utilize apenas o contexto recuperado.
        """
    ),

    MessagesPlaceholder("chat_history"),

    (
        "human",
        """
        Contexto:

        {context}

        Pergunta:

        {question}
        """
    )
])

# CHAIN
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(
            lambda _: history.messages
        )
    }
    | prompt
    | llm
    | StrOutputParser()
)

# CHAT
print("\nChat com PDF iniciado.")
print("Digite 'sair' para encerrar.\n")

while True:

    pergunta = input("Você: ")

    if pergunta.lower() == "sair":
        break

    resposta = rag_chain.invoke(
        pergunta
    )

    print("\nIA:")
    print(resposta)

    history.add_user_message(
        pergunta
    )

    history.add_ai_message(
        resposta
    )

    print("\n" + "=" * 50 + "\n")