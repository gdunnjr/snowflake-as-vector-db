"""
Example of using Snowflake as a vector database. Embeddings are stored in a vector column and snowflake is used in a custom retriever for a langchain retrieval QA chain.

A sample PDF document is loaded, split, embedded and stored in Snowflake. Questions are then asked of it and it gives responses based on 
the data in Snowflake 
"""
import json
from datetime import datetime
import os
import pandas as pd
from langchain_core.documents import Document
from sf_helper_browser_authentication import *
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun 
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# set up values for the embeddings. this example uses azure but you can use openai as well
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-4-32k"
API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT = "gpt-4-32k"
AZURE_API_BASE = "<your_azure_endpoint>"
AZURE_API_VERSION = "2023-07-01-preview"
MAX_TOKENS = 9000

SNOWFLAKE_TABLE_NAME = "LLM_PDF_EXAMPLE"
SNOWFLAKE_HELPER = SnowflakeHelper()

class CustomSnowflakeRetriever(BaseRetriever):
    """
    This class is a custom langchain retriever that uses Snowflake to retrieve documents.
        
        Args:
            snowflake_table_name (str): The name of the Snowflake table.
                
        Returns:
            CustomSnowflakeRetriever: The custom Snowflake retriever.

    """
    snowflake_table_name: str

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # get the embedding for the query
        embedded_question = create_embedding(query)

        # build the sql
        sql =  """
        SELECT content, source, VECTOR_COSINE_SIMILARITY(
            content_vector,
            {embedded_question}::VECTOR(FLOAT,1536)
          ) AS similarity
        FROM {table_name}
        WHERE SIMILARITY >= .5
        ORDER BY SIMILARITY DESC
        """
        sql = sql.replace("{embedded_question}", str(embedded_question)).replace("{table_name}", self.snowflake_table_name)

        # execute the query
        df_answer = SNOWFLAKE_HELPER.execute_sf_query_to_df(sql)

        # convert the data frame to a list of documents
        documents = []
        for index, row in df_answer.iterrows():
            doc = Document(page_content=row["CONTENT"])
            doc.metadata = {
                "source": row["SOURCE"]
            }
            documents.append(doc)
        
        return documents


def load_and_split_pdf(file_path):
    """
    This function loads and splits a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        list: The list of documents.
    """

    loader = PyPDFLoader(file_path)
    docs = loader.load_and_split()

    return docs


def send_docs_to_snowflake(docs, table_name):
    """
    This function sends documents to Snowflake.

    Args:
        docs (list): The documents to send to Snowflake.

    Returns:
        Snowflake: The Snowflake object.
    """
    
    truncate_sf_vector_table(table_name)

    # iterate through the documents and send them to Snowflake
    for doc in docs:
        print("Sending document to Snowflake: " + str(doc.metadata))
        # build the source and metadata fields
        content = doc.page_content
        source = doc.metadata['source']
        metadata={
            'source': source,
            'page': doc.metadata['page']
        }

        # get the embedding for the text
        embedding = create_embedding(content)

        # save to snowflake
        base_sql = """
            INSERT INTO {table_name} 
                (CONTENT, CONTENT_VECTOR, METADATA, SOURCE,  CREATE_DT) 
            SELECT 
                '{content}', {content_vector}::VECTOR(FLOAT,1536), '{metadata}', '{source}', '{create_dt}';
        """ 

        sql = base_sql. \
            replace("{table_name}", table_name). \
            replace("{content}", content). \
            replace("{content_vector}", str(embedding)). \
            replace("{metadata}", json.dumps(metadata)). \
            replace("{source}", source). \
            replace("{create_dt}", datetime.today(). strftime("%m/%d/%Y"))

        SNOWFLAKE_HELPER.execute_sf_query(sql)

    return


def create_embedding(doc_text):
    embeddings: AzureOpenAIEmbeddings = \
        AzureOpenAIEmbeddings(model=EMBEDDING_MODEL,
                              chunk_size=10000,
                              openai_api_key=API_KEY,
                              azure_endpoint=AZURE_API_BASE)
    embedding = embeddings.embed_query(doc_text)

    return embedding


def truncate_sf_vector_table(table_name):
    """
    This function truncates the table in Snowflake that stores the vector data.
    """
    sql = f'TRUNCATE TABLE {table_name};'
    SNOWFLAKE_HELPER.execute_sf_query(sql)

    return


def similarity_search_test(query):
    """
    This function performs a lookup using cosine similarity to find similar OKRs.

    Args:
        query (str): The query string.

    Returns:
        pandas.DataFrame: The dataframe containing the similar OKRs.
    """
    embedded_question = create_embedding(query)
    sql =  """
    SELECT content, VECTOR_COSINE_SIMILARITY(
        content_vector,
        {embedded_question}::VECTOR(FLOAT,1536)
      ) AS similarity
    FROM {SNOWFLAKE_TABLE_NAME}
    WHERE SIMILARITY >= .50
    ORDER BY SIMILARITY DESC
    """
    sql = sql.replace("{embedded_question}", str(embedded_question)).replace("{SNOWFLAKE_TABLE_NAME}", SNOWFLAKE_TABLE_NAME)

    df_answer = SNOWFLAKE_HELPER.execute_sf_query_to_df(sql)

    print('Cosine similarity results:')
    print(df_answer.head())
    print()

    return df_answer


def get_llm():
    """
    This function sets up the LLM for the chat model.
        
        Returns:
            llm: The LLM for the chat model.
    """
    from langchain_openai import AzureChatOpenAI
    llm = AzureChatOpenAI(deployment_name=AZURE_CHAT_DEPLOYMENT, model_name=CHAT_MODEL, temperature=0,
                          openai_api_version=AZURE_API_VERSION, azure_endpoint=AZURE_API_BASE,
                          openai_api_key=API_KEY, streaming=True, max_tokens=MAX_TOKENS)
    return llm


def get_answer(question):
    """
        This function tests a question and answer using the LLM and the custom retriever.
        
        Args:
            question (str): The question to test.
        Returns:
            dict: The answer to the question. 
    """
    retriever = CustomSnowflakeRetriever(snowflake_table_name=SNOWFLAKE_TABLE_NAME)

    # Build prompt
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:"""

    qa_chain_prompt = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(), 
        chain_type='stuff', 
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={"prompt": qa_chain_prompt}
    )

    result = qa_chain.invoke({"query": question})
    return result


# ingest the PDF file into Snowflake
docs = load_and_split_pdf("data/state_of_the_union.pdf")
send_docs_to_snowflake(docs, SNOWFLAKE_TABLE_NAME)

question = "What is going on regarding infrastructure?"

# Test a similarity search from snowflake. Call a function that embeds the query and then uses cosine similarity to find matches in the PDF stored in Snowflake.
similarity_search_test(question)

# test a question and answer
answer = get_answer(question)
print(answer)


