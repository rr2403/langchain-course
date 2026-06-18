import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate #for sending a structured prompt to RAG
from langchain_core.messages import HumanMessage #for invoking the pipeline
from langchain_openai import ChatOpenAI, OpenAIEmbeddings #for embeddings models
from langchain_pinecone import PineconeVectorStore #VECTOR DB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

print("Initializing components... ")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")

vectorstore = PineconeVectorStore(
    index_name = os.environ["INDEX_NAME"], embedding= embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

#Augmentation Part
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context

    {context}

    Question: {question} 

    Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats lines,  and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No. built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error prone
    """

    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    #Step2 : Format documents into context strings
    context = format_docs(docs)

    messages = prompt_template.format_messages(context=context, question = query)
    #print(messages)

    response = llm.invoke(messages)

    return response.content


def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__=="__main__":
    print("Retrieving")

    query= "What is Pinecone in Machine Learning?"
    #Option 0: Raw invocation without RAG

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: RAW LLM Invocation (NO RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\n Answer")
    print(result_raw.content)

    #Option 1: Use implementation WITHOUT LCEL
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: WITHOUT LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\n Answer:")
    print(result_without_lcel)

    
    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)