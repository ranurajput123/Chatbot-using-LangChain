from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://sites.google.com/view/ranuchouhan")
data = loader.load()

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1200, chunk_overlap = 50)
text_chunks = text_splitter.split_documents(data)

from langchain.embeddings import HuggingFaceBgeEmbeddings
embeddings = HuggingFaceBgeEmbeddings()

from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(text_chunks,embeddings)
save_directory = "Storage"
db.save_local(save_directory)
save_directory = "Storage"
new_db = FAISS.load_local(save_directory,embeddings,allow_dangerous_deserialization=True)
retriever = new_db.as_retriever(search_type="mmr",search_kwargs={"k": 3})

from langchain_groq import ChatGroq
groq_api_key = "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"
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=groq_api_key
    # other params...
)

from langchain_core.prompts import ChatPromptTemplate
template = """You are an AI assistant for helping the customers.
            Use the following pieces of retrieved context to answer the question.
            If the question is out of context, just say that you don't know the answer.
            Don't fetch answers from outside the context.
         
            Question:
            {question}
            Context:
            {context}
            Answer:
         """

prompt = ChatPromptTemplate.from_template(template)
from langchain.chains import RetrievalQA
rag_chain = RetrievalQA.from_chain_type(llm,retriever=retriever,chain_type_kwargs={"prompt":prompt})
def process_question(user_question):
    response = rag_chain.invoke(user_question)
    full_response = response["result"]
    return full_response
import gradio as gr
interface = gr.Interface(fn=process_question,
                         inputs=gr.Textbox(lines=2,placeholder="Type your question here"),
                         outputs=gr.Textbox(),
                         title="An AI Assistant",
                         description="Ask any question about documents")
interface.launch(share_server_address=True)