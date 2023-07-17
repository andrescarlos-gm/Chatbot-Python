#importa todas las librerias necesarias

from llama_index import LLMPredictor, PromptHelper
from llama_index import ServiceContext,  GPTListIndex
from llama_index import VectorStoreIndex, SimpleDirectoryReader
import gradio as gr
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain import OpenAI
import openai

#define la llave de la api como variable de entorno

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

#lee la carpeta datos
docs = SimpleDirectoryReader("datos").load_data()

#parametros

max_input_size = 4096
num_outputs = 512
max_chunk_overlap = 20
chunk_size_limit = 600
tokens = 256
chunk_overlap_ratio = 0.2

prompt_Helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio=chunk_overlap_ratio )
modelo = LLMPredictor(llm=OpenAI(temperature=1,model_name="text-davinci-003", max_tokens=tokens))
contexto = ServiceContext.from_defaults(llm_predictor=modelo, prompt_helper=prompt_Helper)

index_model = VectorStoreIndex.from_documents(docs, service_context=contexto)
index_model.storage_context.persist(persist_dir="datos")

def chatbot(input_text):
    modelo = VectorStoreIndex.from_documents(docs)
    query_engine = modelo.as_query_engine()
    streaming_response = query_engine.query(input_text)
    return streaming_response.response

app = gr.Interface(fn=chatbot,
                   inputs= gr.inputs.Textbox(lines=5, label="Escribe tu pregunta aquí..."),
                   outputs= "text",
                   title="Chatbot")
app.launch(share=False)