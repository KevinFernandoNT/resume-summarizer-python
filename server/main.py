from fastapi import FastAPI
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


"""Prompt template used to pass to the model"""
template = """You are an AI helper to summarize and map content given into a specific classification structure. Given some content regarding a person's CV information you will return below items in the given [limitations] as a string array in the defined JSON structure. Only return the JSON and nothing else. 
1. An overall summary of the CV including the person’s professional background [up to 20 words] 
2.What are the education qualification [2-7 words per item] 
3. Work Experience [Include the time of serve for each company] [2-7 words per item]  
4.What are the main skills sets [2-7 words per item]  
  {{ 
    summary: "", 
    education: [], 
    Work experience: [], 
    skills: [], 
  }} 
    CONTENT: 
    {content}"""


"""start a fastapi instance"""
app = FastAPI()


"""make the llm client using langchain ChatOpenAI Class model"""


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        temperature=0,
        openai_api_key="[YOUR OPENAI API TOKEN]",
        max_tokens=3000,
        max_retries=10,
        model="gpt-3.5-turbo",
        request_timeout=600,
    )


"""prepare the prompt by passing the CV content and fromatting the prompt according to the chat model structure"""


def prepare_prompt(content: str) -> ChatPromptTemplate:
    system_message_prompt = SystemMessagePromptTemplate.from_template(template=template)
    chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt])
    return chat_prompt_template.format_prompt(content=content).to_messages()


"""send request openai"""


def make_llm_request(content: str):
    prompt = prepare_prompt(content=content)
    llm = make_llm()
    results = llm(prompt)
    return results.content.strip()


"""root route"""


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


"""route to generate the summary"""


@app.get("/generate-summary")
def read_root():
    """load the pdf file EX :sampleCV.pdf"""
    pdf_loader = PyPDFLoader("./sample.pdf")
    """split the document according to pages when loading for optimization purposes """
    pdf_pages = pdf_loader.load_and_split()

    """split the pages into smaller chunks to prevent exceeding from tokrn count"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=512)
    texts = text_splitter.split_documents(pdf_pages)
    """finally pass the chunk to the llm client to generate a summary"""
    summary = make_llm_request(content=texts[0].page_content)
    try:
        return json.loads(summary)
    except:
        return {"error": "failed to generate the summary"}
