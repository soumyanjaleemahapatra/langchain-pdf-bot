import streamlit as st
from PyPDF2 import PdfReader

#import langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = None

# Configurations
MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"
N_CTX = 2048
N_BATCH = 512

def load_llm_model():
    global llm
    if llm is None:
        try: 
            # Load the LLM model

            # Callbacks support token-wise streaming
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

            # The LLM model used: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf
            # This model needs to be downloaded manually and made available in the root folder of the project.
            # You can download it using the following command:
            # `huggingface-cli download TheBloke/Llama-2-7b-Chat-GGUF llama-2-7b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False`
            # Additional LLM model configuration parameters:
            # - n_ctx: The maximum context size for the model
            # - n_batch: The batch size for the model
            llm = LlamaCpp(
                model_path=MODEL_PATH,
                callback_manager=callback_manager,
                verbose=True,
                n_ctx=N_CTX,
                n_batch=N_BATCH,
            )
        except Exception as e:
            st.error(f"Error loading LLM model: {e}")
            llm = None  # Reset llm to None in case of an error
    return llm

def main():
    st.set_page_config(page_title='Ask your PDF')
    st.header("Ask your PDF 💬")

    # Call the function to load the LLM model
    load_llm_model() 

    # Uploading pdf
    pdf = st.file_uploader('Upload your PDF', type='pdf')
    
    # Reading pdf
    if pdf is not None:
        with st.spinner("Processing PDF..."):
            pdf_reader = PdfReader(pdf)
            # Extracting text from pdf
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap = 200,
            length_function = len
        )    
        chunks = text_splitter.split_text(text)

        # Create embeddings for the chunks
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
         # Create in-memory Qdrant instance
        knowledge_base = Qdrant.from_texts(
            chunks,
            embeddings,
            location=":memory:",
            collection_name="doc_chunks",
        )

        # Show user input
        user_question = st.text_input("Ask a question about your pdf")
        if user_question:
            with st.spinner("Searching for answers..."):
                docs = knowledge_base.similarity_search(user_question)
                chain = load_qa_chain(llm,chain_type="stuff")
                response = chain.run(input_documents = docs, question= user_question)
            # Display response
            st.write(response)


if __name__ == '__main__':
    main()