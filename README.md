# IntraIntel.ai - Multi-LLM Agent Coding Challenge

This project implements a Multi-LLM Agent system in Python designed to answer medical questions. It leverages a custom Model Context Protocol (MCP) to interact with distinct tool servers for information retrieval (Web Search and PubMed Search). The core agent orchestrates multiple free Large Language Models (LLMs) from the Hugging Face Inference API to perform tasks such as query refinement, context snippet summarization, and final answer synthesis using a Retrieval-Augmented Generation (RAG) approach. The system is designed to provide separate, synthesized answers based on the context retrieved from Web Search and PubMed, including links to the original source materials.
## System Architecture

Here’s a high-level flowchart of how the agent processes medical queries:


![Working Flowchart](https://github.com/user-attachments/assets/eccde8f8-f6ea-4e9a-954a-65e71c91ee72)

Here’s a high-level flowchart of how the agent processes medical queries:

Workflow Stages:
User Interaction & Initial Query:
(1. Medical Question): The process starts with the User submitting a medical question.
This question is received by the Main Agent.
Query Processing & Initial Retrieval (Agent Core Logic):
(2. Refined Question): The Main Agent sends the original question to LLM1 (Snippet/Query Refinement) which is hosted on Hugging Face and uses a model like Mistral-Instruct. This step aims to create a more effective search query.
(3. Original Query / Refined Query to MCP Servers): The Main Agent then uses either the original or the refined query to interact with the MCP (Model Context Protocol) servers.
(4. Web Search Query): The query is sent to the MCP Server: Web Search (implemented in pub_search_server.py - Note: diagram text says pub_search_server.py here, but it should likely be web_search_server.py for clarity).
This server queries DuekDukGo (DuckDuckGo), which in turn searches Infenet Web Pages (Internet Web Pages).
(4b. Formatted Web Results): The Web Search MCP server returns formatted results (likely JSON) to the Main Agent, which are then passed to the Final Output stage and also used for further processing.
The Main Agent also sends the query to the NCB PubMed (NCBI PubMed) MCP server (implemented in pubmed.search.server.py).
This server queries the PubMed Database.
(Diagram note: The return path from PubMed MCP to Main Agent seems to be missing an explicit "formatted results" arrow like 4b, but it's implied it provides data for LLM3 later).
Context Processing & Answer Synthesis (External Services & APIs):
(9. Refined Web Snippets): The refined query/results from LLM1 (labeled "Refined Web Snippets" from LLM1 output) are fed into LLM2 (Answer Synthesis for Web).
(4b. Formatted Oreta Quesign+ (JSON)): This label seems to indicate formatted data, possibly from web search, is also an input to LLM2. (The label "Oreta Quesign+" is unclear and might be a typo or specific internal term).
LLM2 interacts with Hugging Face using a model like DISTILBART-CNN (labeled as LLM3: DISTILBART-CNN on the Hugging Face box, but logically this is the summarizer/synthesizer for web results, so it's acting as the second distinct LLM function in the pipeline if LLM1 was query refinement and this is for web synthesis).
(9a. Synthesized Answer (Web) / Summarized Web Snippets): LLM2 produces a synthesized answer based on web context or summarized snippets.
(6a. Formatted Web Context + Original Question): This data (presumably from the Main Agent after processing web results and potentially after LLM2 if LLM2 was a summarizer) is fed into another Hugging Face instance labeled LLM2: Mistral+Instruct (this is confusing as LLM1 was also Mistral-Instruct. It likely means another call to a Mistral-Instruct type model for final web answer synthesis if LLM2 was purely a summarizer, or it might be a mislabeling if LLM2 itself was the web answer synthesizer).
PubMed Path to Synthesis:
(7b. Synthesized Answer Context / Original Question): Context derived from PubMed results, along with the original question, is sent to LLM3 (Answer Synthesis for PubMed).
This LLM3 interacts with Hugging Face using a model labeled LLM3: Answer Synthesis (model type not specified on this box but implied to be suitable for synthesis, potentially another Mistral-Instruct call).
(11b. Synthesized Answer (PubMed)): LLM3 produces a synthesized answer based on PubMed context.
Final Output:
(10. Final Output (Separate Answers + Sources)): The Main Agent consolidates the synthesized answers from the web path and the PubMed path.
This Final Output, containing separate answers and their respective sources, is presented back to the User.
Diagram Interpretation Notes & Discrepancies from Code:
LLM Naming/Function: The diagram labels LLMs (LLM1, LLM2, LLM3) somewhat ambiguously with respect to their specific roles (snippet refinement vs. query refinement, summarization vs. synthesis for web/PubMed). The code has a clearer three-LLM pipeline: LLM1 (Query Refinement), LLM2 (Snippet Summarization for both sources), and LLM3 (Answer Synthesis for each source separately).
"Agent Gore Logic": This is likely a typo and should be "Agent Core Logic."
MCP Server File Names: The Web Search server is associated with pub_search_server.py in one label, which is likely an error; it should be web_search_server.py. The PubMed server is pubmed.search.server.py.
Label "4b. Formatted Oreta Quesign+ (JSON)": This label is unclear. It likely refers to formatted data from the web search being passed to an LLM.
Hugging Face Model Labels:
"LLM2: Mistral+Instruct" is used for the first Hugging Face box.
"LLM3: DISTILBART-CNN" for the second.
"LLM3: Answer Synthesis" for the third.
This labeling suggests perhaps three types of LLM interactions, but the flow implies specific roles that might be better aligned with the code's LLM1 (Refine), LLM2 (Summarize), LLM3 (Synthesize) structure. For example, DISTILBART-CNN is typically a summarization model. Mistral-Instruct is an instruction-following model good for refinement or synthesis.
Simplified Interpretation for README based on the Diagram's Visual Flow (acknowledging potential label inconsistencies):
The diagram shows that the user's medical question initiates a process within the Main Agent. This agent may refine the question using a first LLM (LLM1 via Hugging Face). The query is then sent to MCP servers for Web Search (using DuckDuckGo) and PubMed Search. The results from web search are processed, potentially by another LLM (LLM2, possibly for summarization or initial web-based answer synthesis, using a model like DistilBERT-CNN via Hugging Face). A separate LLM (LLM3, potentially Mistral-Instruct via Hugging Face) takes context from PubMed to generate a PubMed-specific answer. The Main Agent then combines these outputs into a final response for the user, presenting information derived from different sources.

## Project Structure
Intra_intel_multi_llm_challenge/
├── mcp_servers/
│ ├── init.py
│ ├── web_search_server.py # MCP server for general web search (uses DuckDuckGo)
│ └── pubmed_search_server.py # (Bonus) MCP server for PubMed search (uses NCBI Entrez)
├── agent/
│ ├── init.py
│ └── main_agent.py # Main agent orchestrator with Multi-LLM pipeline
├── .env.example # Example for environment variables
├── .env # Your environment variables (ignored by git if .gitignore is set up)
├── requirements.txt # Python dependencies
└── README.md # This file (you are here!)

## Core Approach & Design

The system follows a multi-step, agentic RAG pipeline:

1.  **User Input:** The agent takes a medical question from the user.
2.  **(Optional) Query Refinement (LLM1):** The user's question can be passed to a first LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.3`) to generate a more concise and effective search query, focusing on key medical terms.
3.  **Information Retrieval (MCP Tools):**
    *   The (potentially refined) search query is sent to one or both MCP tool servers:
        *   **Web Search MCP Server:** Queries the general web using `duckduckgo-search`.
        *   **PubMed MCP Server:** Queries the PubMed database using `BioPython`'s Entrez utilities.
    *   These servers return structured search results (title, snippet, URL/ID).
4.  **Context Processing:**
    *   **(Optional) Snippet Summarization (LLM2):** For each search result, if the snippet is long, it can be passed to a specialized summarization LLM (e.g., `sshleifer/distilbart-cnn-6-6`) to create a more concise version.
    *   **Context Formatting:** The (summarized or original) snippets from each source (Web Search, PubMed) are formatted separately into structured context strings. Each item in the context is labeled (e.g., "Item 1", "Item 2") and includes its title, content, and reference (URL or PubMed ID).
5.  **Answer Synthesis (LLM3 - RAG):**
    *   For each information source (Web Search, PubMed), the agent makes a separate call to a powerful instruction-tuned LLM (e.g., `mistralai/Mistral-7B-Instruct-v0.3`).
    *   The prompt to this LLM includes:
        *   The original user question.
        *   The formatted context *from that specific source*.
        *   An instruction to answer based *only* on the provided context and to cite item numbers if possible.
    *   This LLM generates a synthesized answer for that source.
6.  **Output:** The system presents the original question, followed by the synthesized answer derived from Web Search (with source links) and then the synthesized answer derived from PubMed (with source links).

**Key Technologies:**
*   **Python 3.9+** with `asyncio` for concurrency.
*   **FastAPI & Uvicorn:** For building the asynchronous MCP tool servers.
*   **`httpx`:** For asynchronous HTTP requests from the agent to MCP servers and the Hugging Face API.
*   **Hugging Face Inference API:** To access free LLMs for refinement, summarization, and synthesis.
*   **`duckduckgo-search`:** For API-key-free web searches.
*   **`BioPython`:** For querying NCBI's PubMed database.
*   **`python-dotenv`:** For managing API keys and environment variables.

## Model Context Protocol (MCP) Definition

The MCP used in this project is a simple HTTP-based JSON protocol for communication between the main agent and the tool servers:

*   **Request to MCP Server:**
    *   **Endpoint:** `POST /execute`
    *   **JSON Body:** `{"query": "user's search query string"}`
*   **Response from MCP Server:**
    *   **JSON Body (Success):**
        ```json
        {
          "source": "server_name_string (e.g., WebSearchServer, PubMedServer)",
          "status": "success",
          "results": [
            {
              "title": "string",
              "snippet": "string",
              "url": "string_url_if_web_search", // or missing
              "id": "string_pubmed_id_if_pubmed" // or missing
            }
            // ... more results
          ]
        }
        ```
    *   **JSON Body (Error):**
        ```json
        {
          "source": "server_name_string",
          "status": "error",
          "error_message": "string_describing_the_error"
        }
        ```

## Setup Instructions

1.  **Clone/Download:**
    *   Obtain the project files and place them in a directory named `intra_intel_multi_llm_challenge`.

2.  **Create and Activate Python Virtual Environment:**
    *   Navigate to the project root directory (`intra_intel_multi_llm_challenge`).
    *   Create the environment:
        ```bash
        python -m venv venv
        ```
    *   Activate it:
        *   On macOS/Linux: `source venv/bin/activate`
        *   On Windows: `venv\Scripts\activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   In the project root, copy the file `.env.example` to a new file named `.env`.
    *   Open the `.env` file and fill in your details:
        ```env
        # .env
        HF_API_TOKEN="your_huggingface_api_token_here"
        NCBI_EMAIL="your_email_for_pubmed@example.com"
        ```
        *   **`HF_API_TOKEN`:** Get this from your Hugging Face account settings ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). A token with `read` permissions is sufficient.
        *   **`NCBI_EMAIL`:** Provide your email address for polite programmatic access to NCBI's PubMed database via Entrez. No prior registration of this email with NCBI is needed for this project.

5.  **Accept Terms for Gated Models (Crucial for Mistral):**
    *   The default text generation LLM used is `mistralai/Mistral-7B-Instruct-v0.3`. This is a gated model.
    *   **Log in** to your Hugging Face account on their website.
    *   Go to the model card page: [https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
    *   Read and **accept the license terms** to gain access to this model. Your `HF_API_TOKEN` will then be authorized to use it. If you skip this, API calls to this model will fail.

## Running the System

The system requires the MCP servers to be running before the main agent can query them.

1.  **Terminal 1: Start the Web Search MCP Server:**
    *   Navigate to the project root (`intra_intel_multi_llm_challenge`).
    *   Ensure your virtual environment is activated.
    *   Run:
        ```bash
        python -m uvicorn mcp_servers.web_search_server:app --reload --port 8001
        ```
    *   Wait for the message indicating Uvicorn is running (e.g., `Uvicorn running on http://0.0.0.0:8001`).

2.  **Terminal 2: Start the PubMed MCP Server (for Bonus Functionality):**
    *   Navigate to the project root.
    *   Ensure your virtual environment is activated.
    *   Run:
        ```bash
        python -m uvicorn mcp_servers.pubmed_search_server:app --reload --port 8002
        ```
    *   Wait for Uvicorn startup confirmation.

3.  **Terminal 3: Run the Main Agent:**
    *   Navigate to the project root.
    *   Ensure your virtual environment is activated.
    *   Run:
        ```bash
        python agent/main_agent.py
        ```
    *   The agent will process a predefined list of 5 medical questions. For each question, it will:
        *   Optionally refine the query using LLM1.
        *   Query the Web Search MCP server.
        *   Optionally summarize web search snippets using LLM2.
        *   Synthesize an answer based on web search context using LLM3.
        *   Print the web search based answer and its source links.
        *   Query the PubMed MCP server (as `use_pubmed_for_this_question` is `True` by default in examples).
        *   Optionally summarize PubMed snippets using LLM2.
        *   Synthesize an answer based on PubMed context using LLM3.
        *   Print the PubMed based answer and its source links.
    *   There will be pauses between processing each question to be considerate to the Hugging Face API rate limits.

    **Note on Hugging Face API Model Loading:** The first time an LLM is called via the Inference API, it might need to be loaded by Hugging Face, which can take 20-90 seconds (or more for larger models like Mistral-7B). Subsequent calls are typically faster. The script includes timeouts and attempts to handle model loading messages.

## Customization and Debugging

*   **LLM Choices:** You can change `VERIFIED_INSTRUCT_MODEL_ID` and `SUMMARIZATION_MODEL_ID` in `agent/main_agent.py` if you find other models that work better or are more reliably available on the Hugging Face free inference API (always verify with the widget on the model's card page first!).
*   **Toggle Features:** In the `if __name__ == "__main__":` block of `agent/main_agent.py`, you can change the `refine_queries` and `summarize_snippets` boolean flags passed to `main_agent_workflow` to test the pipeline with these steps enabled or disabled. You can also change `use_pubmed_for_this_question` for individual calls.
*   **Verbose Logging:** To see detailed debug information (API payloads, full contexts, etc.):
    1.  Open `agent/main_agent.py`.
    2.  Change `LOG_LEVEL = logging.INFO` to `LOG_LEVEL = logging.DEBUG`.
    3.  Save and re-run the agent.

## Dependencies

The main dependencies are listed in `requirements.txt`:
*   `fastapi`: For building the MCP servers.
*   `uvicorn[standard]`: ASGI server for FastAPI.
*   `httpx`: For making asynchronous HTTP requests.
*   `duckduckgo-search`: For web search functionality.
*   `biopython`: For querying PubMed via NCBI Entrez.
*   `python-dotenv`: For managing environment variables.

This README should cover the necessary details for understanding, setting up, and running the project.

## Sample medical questions output screenshot

Web search
![Web search output](https://github.com/user-attachments/assets/8fe4be5b-df27-439f-bdef-95394f53edab)

Pubmed search
![PubmedSearch Output](https://github.com/user-attachments/assets/e941063a-d7b9-4920-bea9-4e99e44d1fd3)
