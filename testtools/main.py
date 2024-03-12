import os
import argparse

from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, Document

import chromadb

from llama_index.llms.openai import OpenAI


embed_model = OpenAIEmbedding()
# create client and a new collection
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

# set up ChromaVectorStore and load in data
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Set up defaults and get API key from environment variable
defaults = {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "inputs": ".",
}

llm = OpenAI(
    api_key=defaults["api_key"],
)


# Function to validate and parse arguments
def validate_and_parse_args(parser):
    args = parser.parse_args()

    for key, value in vars(args).items():
        if not value:
            args.__dict__[key] = parser.get_default(key)

    if not args.api_key:
        parser.error(
            "The --api-key argument is required if OPENAI_API_KEY environment variable is not set."
        )
    if not args.prompt:
        parser.error("The --prompt argument is required.")

    return args


def main():
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="RAG")
    parser.add_argument(
        "-k",
        "--api-key",
        type=str,
        default=defaults["api_key"],
        help="OpenAI API key. Can also be set with OPENAI_API_KEY environment variable.",
    )
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Prompt.")
    parser.add_argument(
        "-i",
        "--inputs",
        type=str,
        default=defaults["inputs"],
        help="Comma separated list of input files or directories.",
    )

    args = validate_and_parse_args(parser)

    input_files = args.inputs.split(",")

    ds = [d for d in input_files if os.path.isdir(d)]
    fs = [f for f in input_files if os.path.isfile(f)]
    invalid = [i for i in input_files if i not in ds and i not in fs]

    if invalid:
        raise Exception(f"Invalid input files or directories: {', '.join(invalid)}")

    documents: list[Document] = []
    if ds:
        for d in ds:
            reader = SimpleDirectoryReader(
                input_dir=d,
                recursive=True,
                exclude_hidden=False,
            )
            documents.extend(reader.load_data(show_progress=True))

    if fs:
        reader = SimpleDirectoryReader(
            input_files=fs,
            exclude_hidden=False,
        )
        documents.extend(reader.load_data(show_progress=True))

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embedding=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=2)

    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    response = query_engine.query(args.prompt)

    print(response)


if __name__ == "__main__":
    main()
