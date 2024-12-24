
import os
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone
import json
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_KEY = os.environ.get('HACKATHON_API_KEY')
PINECONE_KEY = os.environ.get('ANDREW_API_KEY')
PINECONE_INDEX = os.environ.get('RAG_PINECONE_INDEX')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inference(input_text, temperature: float = 0.7):

    system_prompt = '''
    You are an expert AI assistant specializing in Ehlers-Danlos syndrome (EDS). Your role is to provide comprehensive, accurate, and well-structured answers about EDS. Try to give your response in an elegant Markdown format and in multiple paragraphs. Follow these guidelines:

    1. In the first paragraph, begin with a broad overview that directly addresses the main question.
    2. In the second paragraph, provide detailed information mainly by using the given Context. Also use your trained knowledge about EDS to supplement the assertions. Aim for a balance between these sources.
    3. Answer in multiple paragraphs and be comprehensive in your answer
    4. Structure your response logically:
       a) Start with a general answer to the question.
       b) Provide specific examples or details, always with proper citations (use the provided references marked as '(Ref: ').
       c) If relevant, mention any contradictions or areas of ongoing research.
    5. If mentioning specific studies or cases, clearly state their relevance to the main question and provide proper context.
    6. In the last paragraph, conclude with a brief summary of the key points, if the answer is lengthy.    
    IMPORTANT: If you receive a question unrelated to Ehlers-Danlos Syndrome (EDS), respond directly by stating that the question is not related, without providing any additional context or explanations. For example, if the question is "Who is the actor in the movie titanic" and even if it has any EDS context given in the "Context", your answer should be like "Sorry, this question is not related to EDS and I cannot address that."
    
    '''

    if not input_text:
        logging.error("No input text provided")
        return json.dumps({"error": "No input text provided"}), 400
    try:
        rag_context = get_rag_context(input_text)

        instruction_prompt = f'''
        User message: {input_text}
        Context : {rag_context}
        Always make sure to provide references in your answer. You can find the references in the Context marked as '(Ref: '.
        '''
        response = get_groq_response(instruction_prompt, system_prompt, "llama3-8b-8192", temperature=temperature)
        
        return json.dumps(response)
    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        return json.dumps({"error": str(ve)}), 500
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return json.dumps({"error": str(e)}), 500



def get_rag_context(query, top_k=5):
    try:
        embed_model = OpenAIEmbedding(
            model='text-embedding-ada-002',
            api_key=OPENAI_KEY,
        )
        Settings.embed_model = embed_model
        pc = Pinecone(api_key=PINECONE_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX)
        query_embedding = embed_model.get_text_embedding(query)
        retrieved_doc = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        # extracted_context_summary = list(map(lambda x: json.loads(x.metadata['_node_content'])['metadata']['section_summary'], retrieved_doc.matches))
        extracted_context_summary = list(map(lambda x: json.loads(x.metadata['_node_content'])['metadata']['text'], retrieved_doc.matches))
        provenance = list(map(lambda x: x.metadata['c_document_id'], retrieved_doc.matches))
        context = ''
        for i in range(top_k):
            context += extracted_context_summary[i] + '(Ref: ' + provenance[i] + '). '
        return context
    except Exception as e:
        logging.error(f"Failed to retrieve or process context: {str(e)}")
        return "Error retrieving context."




def get_groq_response(instruction, system_prompt, chat_model, temperature=0.3):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in the environment variables")
    groq_client = Groq(
        api_key=GROQ_API_KEY,
    )
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        model=chat_model,
        temperature=temperature
    )
    if chat_completion.choices:
        return chat_completion.choices[0].message.content
    else:
        return 'Unexpected response'