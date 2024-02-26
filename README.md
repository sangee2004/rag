# RAG

Retrieval-Augmented Generation for GPTScript.
Leveraging an embedding model and a generation model behind OpenAI API, the RAG tool can answer prompts based on provided documents.
There is an adhoc mode where nothing is persisted and we're using an in-memory vector database, so the embeddings don't persist between runs.

## Preqrequisites

- Python 3.10+
- OpenAI API Key - exported as `OPENAI_API_KEY` environment variable

## Usage

```bash
gptscript tool.gpt --prompt "<your question>" --inputs "<your documents>"`
```

### CLI Arguments

- `--prompt` - The prompt to ask the model
- `--inputs` - The documents to use for retrieval: comma-separated list of files or directories

### Examples

#### Check this README file and ask about the CLI options for the RAG tool

```bash
gptscript tool.gpt --prompt "What are the CLI options for the RAG tool?" --inputs "README.md"
```

is the same as

```bash
gptscript examples/readme.gpt
```
