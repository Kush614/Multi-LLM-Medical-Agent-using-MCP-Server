# agent/main_agent.py
import httpx
import asyncio
import os
from dotenv import load_dotenv
import logging
import json

# --- Logging Configuration ---
LOG_LEVEL = logging.INFO 
# LOG_LEVEL = logging.DEBUG 
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MultiLLMAgent")

# --- Environment Variable Loading & Config (same as before) ---
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_API_TOKEN: logger.error("HF_API_TOKEN not found."); raise ValueError("HF_API_TOKEN not found.")
WEB_SEARCH_MCP_URL = "http://localhost:8001/execute"
PUBMED_MCP_URL = "http://localhost:8002/execute"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"}
VERIFIED_INSTRUCT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
QUERY_REFINEMENT_MODEL_ID = VERIFIED_INSTRUCT_MODEL_ID
QUERY_REFINEMENT_API_URL = f"https://api-inference.huggingface.co/models/{QUERY_REFINEMENT_MODEL_ID}"
SUMMARIZATION_MODEL_ID = "sshleifer/distilbart-cnn-6-6" 
SUMMARIZATION_API_URL = f"https://api-inference.huggingface.co/models/{SUMMARIZATION_MODEL_ID}"
SYNTHESIS_MODEL_ID = VERIFIED_INSTRUCT_MODEL_ID
SYNTHESIS_API_URL = f"https://api-inference.huggingface.co/models/{SYNTHESIS_MODEL_ID}"

# --- HF Inference Request (same as previous) ---
async def hf_inference_request(api_url: str, payload: dict, client_session: httpx.AsyncClient, model_name_for_log: str) -> dict | list | None:
    # ... (exact same code as the last fully commented version)
    try:
        logger.debug(f"Sending request to HF API ({model_name_for_log} at {api_url}) with payload keys: {list(payload.keys())}")
        response = await client_session.post(api_url, headers=HF_HEADERS, json=payload, timeout=90.0)
        if response.status_code == 200: response_data = response.json(); logger.debug(f"HF API ({model_name_for_log}): Success."); return response_data
        else:
            error_content_bytes = await response.aread(); error_content = error_content_bytes.decode(errors='replace')
            error_message = f"Unknown error (Status {response.status_code})"; error_details_dict = {"error": error_message, "model": model_name_for_log}
            try:
                error_json = json.loads(error_content); error_message = error_json.get("error", str(error_json))
                if isinstance(error_message, list): error_message = " ".join(error_message)
                error_details_dict["error"] = f"HF API Error (Status {response.status_code}): {error_message}"
                estimated_time = error_json.get("estimated_time")
                if estimated_time: logger.warning(f"HF API ({model_name_for_log}) model loading, est_time: {estimated_time:.2f}s."); error_details_dict["error"] = "model_loading"; error_details_dict["estimated_time"] = estimated_time
            except json.JSONDecodeError: error_details_dict["error"] = f"HF API Error (Status {response.status_code}): {error_content if error_content else 'No error body'}"
            logger.error(f"Error HF API ({model_name_for_log}): Status {response.status_code}, Resp: {error_details_dict['error']}")
            return error_details_dict
    except httpx.TimeoutException: logger.error(f"Timeout HF API ({model_name_for_log})."); return {"error": f"Timeout for {model_name_for_log}.", "model": model_name_for_log}
    except Exception as e: logger.error(f"General Error HF API ({model_name_for_log}): {e}", exc_info=(LOG_LEVEL == logging.DEBUG)); return {"error": f"General Error: {str(e)}", "model": model_name_for_log}


# --- LLM1: Query Refinement (same as previous) ---
async def refine_query_with_llm1(original_query: str, client_session: httpx.AsyncClient) -> str:
    # ... (exact same code as the last fully commented version)
    logger.info(f"LLM1 ({QUERY_REFINEMENT_MODEL_ID}): Refining query..."); logger.debug(f"Original query for LLM1: '{original_query}'")
    prompt = (f"Generate a concise and effective search query for medical databases based on the "
              f"following user question. Focus on key medical terms, conditions, and treatments. "
              f"Output only the search query itself, without any preamble. "
              f"User question: \"{original_query}\"\nSearch query:")
    payload = { "inputs": prompt, "parameters": {"max_new_tokens": 30, "do_sample": False, "temperature": 0.1, "return_full_text": False}, "options": {"wait_for_model": True, "use_cache": False} }
    if "return_full_text" not in payload["parameters"]: payload["parameters"]["return_full_text"] = True 
    response_data = await hf_inference_request(QUERY_REFINEMENT_API_URL, payload, client_session, f"LLM1_Refine({QUERY_REFINEMENT_MODEL_ID})")
    if response_data and not (isinstance(response_data, dict) and response_data.get("error")) and isinstance(response_data, list) and len(response_data) > 0 and isinstance(response_data[0], dict) and "generated_text" in response_data[0]:
        generated_output = response_data[0]["generated_text"]; refined_query = generated_output.strip()
        if payload["parameters"].get("return_full_text", True) and "\nSearch query:" in prompt: parts = generated_output.split("\nSearch query:"); refined_query = parts[-1].strip() if len(parts) > 1 else generated_output.strip()
        refined_query = refined_query.split('\n')[0].strip().replace("\"", "").replace("Search query:", "").strip()
        if refined_query and len(refined_query) >= 3: logger.info(f"LLM1: Refined query to: '{refined_query}'"); return refined_query
        logger.warning(f"LLM1: Refined query too short ('{refined_query}'). Using original."); return original_query
    logger.warning(f"LLM1: Failed to refine. Using original. Details: {str(response_data)[:100] if response_data else 'No response'}")
    return original_query

# --- LLM2: Summarization (same as previous) ---
async def summarize_text_with_llm2(text_to_summarize: str, client_session: httpx.AsyncClient, max_length=60, min_length=15) -> str:
    # ... (exact same code as the last fully commented version)
    if not text_to_summarize or len(text_to_summarize.split()) < min_length + 10: return text_to_summarize
    logger.info(f"LLM2 ({SUMMARIZATION_MODEL_ID}): Summarizing snippet..."); logger.debug(f"LLM2 Summarizing (first 50): '{text_to_summarize[:50]}...'")
    max_input_chars = 1024; payload = { "inputs": text_to_summarize[:max_input_chars], "parameters": {"max_length": max_length, "min_length": min_length, "do_sample": False}, "options": {"wait_for_model": True, "use_cache": False} }
    if len(text_to_summarize) > max_input_chars: logger.warning(f"LLM2: Input for summarization truncated.")
    response_data = await hf_inference_request(SUMMARIZATION_API_URL, payload, client_session, f"LLM2_Summarize({SUMMARIZATION_MODEL_ID})")
    if response_data and not (isinstance(response_data, dict) and response_data.get("error")) and isinstance(response_data, list) and len(response_data) > 0 and isinstance(response_data[0], dict) and "summary_text" in response_data[0]:
        summary = response_data[0]["summary_text"].strip(); logger.info(f"LLM2: Summarized snippet."); logger.debug(f"LLM2 Summary: '{summary[:50]}...'"); return summary
    logger.warning(f"LLM2: Failed to summarize. Returning original. Details: {str(response_data)[:100] if response_data else 'No response'}")
    return text_to_summarize

# --- LLM3: Synthesis (Modified Prompt for Citation) ---
async def synthesize_answer_with_llm3(user_query: str, context: str, source_name_for_prompt: str, client_session: httpx.AsyncClient) -> str:
    logger.info(f"LLM3 ({SYNTHESIS_MODEL_ID}): Synthesizing answer for '{source_name_for_prompt}' source...")
    if not context: logger.warning(f"LLM3: No context from {source_name_for_prompt} for synthesis."); return f"No information found from {source_name_for_prompt} to answer the question."
    
    # Modified prompt to encourage citation of item numbers
    prompt_template = (
        f"You are a helpful medical AI assistant. Based ONLY on the provided context from '{source_name_for_prompt}', "
        "answer the user's question. If specific information is drawn from an item in the context, "
        "please indicate which item (e.g., 'According to Item 1,...' or '[Source: Item 2]'). "
        "If the context is insufficient, clearly state that. Do not use any external knowledge.\n\n"
        f"Context from {source_name_for_prompt} (Items are numbered):\n{{context}}\n\n" # Added "(Items are numbered)"
        "User Question: {question}\n\nAnswer:"
    )
    # ... (rest of the synthesize_answer_with_llm3 function is the same as the previous fully commented version) ...
    MAX_INPUT_CHARS = 3800; fixed_len = len(prompt_template.format(context="", question="").replace("{context}","").replace("{question}","")); avail_ctx_q = MAX_INPUT_CHARS - fixed_len; avail_ctx = avail_ctx_q - len(user_query)
    if avail_ctx <= 100: return "Question too long or prompt structure leaves too little space for context."
    if len(context) > avail_ctx: context = context[:avail_ctx - 3] + "..."; logger.warning(f"LLM3: Context for {source_name_for_prompt} truncated.")
    final_input = prompt_template.format(context=context, question=user_query)
    logger.debug(f"LLM3 ({source_name_for_prompt}) final input (first 100): {final_input[:100]}...")
    payload = { "inputs": final_input, "parameters": {"max_new_tokens": 350, "do_sample": True, "top_p": 0.9, "temperature": 0.3, "return_full_text": False}, "options": {"wait_for_model": True, "use_cache": False} }
    if "return_full_text" not in payload["parameters"]: payload["parameters"]["return_full_text"] = True
    response_data = await hf_inference_request(SYNTHESIS_API_URL, payload, client_session, f"LLM3_Synth_{source_name_for_prompt}({SYNTHESIS_MODEL_ID})")
    if response_data and not (isinstance(response_data, dict) and response_data.get("error")) and isinstance(response_data, list) and len(response_data) > 0 and isinstance(response_data[0], dict) and "generated_text" in response_data[0]:
        gen_output = response_data[0]["generated_text"]; answer = gen_output.strip()
        if payload["parameters"].get("return_full_text", True) and "\nAnswer:" in final_input: parts = gen_output.split("\nAnswer:"); answer = parts[-1].strip() if len(parts) > 1 else gen_output.strip()
        answer = answer.split('\n\n')[0].strip()
        logger.info(f"LLM3: Synthesized answer from {source_name_for_prompt} successfully."); return answer if answer else f"AI model generated an empty answer for {source_name_for_prompt}."
    err_msg = response_data.get('error', 'Unknown error') if isinstance(response_data, dict) else 'Malformed response'
    logger.error(f"LLM3: Failed to synthesize for {source_name_for_prompt}. Details: {err_msg}")
    return f"AI model error during synthesis for {source_name_for_prompt}: {err_msg}"


# --- MCP Server Querying (same as previous) ---
async def query_mcp_server(url: str, query: str, client_session: httpx.AsyncClient) -> dict:
    # ... (same code as previous fully commented version) ...
    server_name = "WebSearch" if "8001" in url else "PubMed" if "8002" in url else "UnknownServer"
    logger.info(f"Querying {server_name} with: '{query[:60]}...'")
    payload = {"query": query}
    try:
        response = await client_session.post(url, json=payload, timeout=25.0); response.raise_for_status()
        data = response.json(); logger.info(f"{server_name}: Success, {len(data.get('results',[]))} results.")
        if 'source' not in data: data['source'] = server_name 
        return data
    except Exception as e: logger.error(f"Error querying {server_name} ({url}): {str(e)[:150]}"); return {"source": server_name, "status": "error", "error_message": str(e)}

# --- Context Formatting for a SINGLE Source (Modified to clearly label items) ---
async def format_source_context(mcp_response: dict, client_session: httpx.AsyncClient, enable_summarization: bool) -> str:
    context_str = ""
    if not (isinstance(mcp_response, dict) and mcp_response.get("status") == "success" and isinstance(mcp_response.get("results"), list)):
        logger.warning(f"Skipping problematic/error response from source: {mcp_response.get('source', 'Unknown') if isinstance(mcp_response, dict) else 'Malformed'}")
        return ""

    display_name = mcp_response.get('source', 'UnknownSource') 
    # context_str += f"\n--- Context from {display_name} ---\n" # This header is now part of the LLM3 prompt

    # Process up to top N_CONTEXT_ITEMS results from each source
    N_CONTEXT_ITEMS = 3 # How many items to include in the context for the LLM

    for item_idx, item in enumerate(mcp_response["results"][:N_CONTEXT_ITEMS]): 
        title = item.get('title', 'N/A')
        snippet = item.get('snippet', 'N/A')
        url_or_id = item.get('url') or f"PubMed ID: {item.get('id')}" # Get URL or construct PubMed ID string
        
        snippet_to_use = snippet
        if enable_summarization and snippet and len(snippet.split()) > 40: # Summarize if >40 words
            summarized_snippet = await summarize_text_with_llm2(snippet, client_session)
            snippet_to_use = summarized_snippet if summarized_snippet else snippet
        else: snippet_to_use = (snippet[:250] + '...') if len(snippet) > 250 else snippet # Truncate slightly more

        # Clearly label each item for the LLM to reference
        context_str += f"Item {item_idx + 1}:\n"
        context_str += f"  Title: {title[:100]}\n" # Truncate title for context
        context_str += f"  Content: {snippet_to_use}\n"
        if url_or_id: # Add URL/ID to context if available
             context_str += f"  Reference: {url_or_id}\n"
        context_str += "---\n"
    
    return context_str.strip()


# --- Main Agent Workflow (Modified for printing sources) ---
async def main_agent_workflow(
    original_user_question: str, 
    use_pubmed_for_this_question: bool = True,
    enable_query_refinement: bool = True,
    enable_snippet_summarization: bool = True 
    ):
    print(f"\n\n médica|| Question: \"{original_user_question}\"")
    logger.info(f"--- Starting Workflow for: '{original_user_question}' ---")
    
    async with httpx.AsyncClient() as client_session:
        search_query = original_user_question
        if enable_query_refinement:
            refined_q = await refine_query_with_llm1(original_user_question, client_session)
            if refined_q and refined_q.lower().strip() != original_user_question.lower().strip() and len(refined_q) > 3:
                search_query = refined_q
            await asyncio.sleep(0.2)
        
        # --- Web Search Section ---
        print(f"\n--- Analyzing Web Search Results for: \"{search_query[:70]}{'...' if len(search_query)>70 else ''}\" ---")
        web_search_response_data = await query_mcp_server(WEB_SEARCH_MCP_URL, search_query, client_session)
        web_search_context = await format_source_context(web_search_response_data, client_session, enable_snippet_summarization)
        
        logger.debug(f"--- Web Search Context for LLM3 ---\n{web_search_context}\n-------------------------")
        web_search_answer = await synthesize_answer_with_llm3(original_user_question, web_search_context, "Web Search", client_session)
        
        print(f"\n🤖 Answer based on Web Search:")
        print("-" * 30)
        print(web_search_answer) # LLM3 handles "no context" message
        print("-" * 30)
        if web_search_context and web_search_response_data.get("status") == "success":
            print("  Sources from Web Search:")
            for i, item in enumerate(web_search_response_data.get("results", [])[:3]): # Show top 3 sources used
                print(f"    {i+1}. {item.get('title', 'N/A')}")
                print(f"       Link: {item.get('url', 'N/A')}")
        elif web_search_response_data.get("status") == "error":
            print(f"  Web Search Error: {web_search_response_data.get('error_message', 'Unknown error')}")


        # --- PubMed Search Section (Conditional) ---
        if use_pubmed_for_this_question:
            await asyncio.sleep(1) # Increased pause before PubMed operations
            print(f"\n--- Analyzing PubMed Results for: \"{search_query[:70]}{'...' if len(search_query)>70 else ''}\" ---")
            pubmed_response_data = await query_mcp_server(PUBMED_MCP_URL, search_query, client_session)
            pubmed_context = await format_source_context(pubmed_response_data, client_session, enable_snippet_summarization)

            logger.debug(f"--- PubMed Context for LLM3 ---\n{pubmed_context}\n-------------------------")
            pubmed_answer = await synthesize_answer_with_llm3(original_user_question, pubmed_context, "PubMed", client_session)

            print(f"\n🔬 Answer based on PubMed Search:")
            print("-" * 30)
            print(pubmed_answer) # LLM3 handles "no context" message
            print("-" * 30)
            if pubmed_context and pubmed_response_data.get("status") == "success":
                print("  Sources from PubMed:")
                for i, item in enumerate(pubmed_response_data.get("results", [])[:3]): # Show top 3 sources used
                    pmid = item.get('id', 'N/A')
                    print(f"    {i+1}. {item.get('title', 'N/A')}")
                    print(f"       PubMed ID: {pmid} (Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")
            elif pubmed_response_data.get("status") == "error":
                print(f"  PubMed Search Error: {pubmed_response_data.get('error_message', 'Unknown error')}")
        
        logger.info(f"--- Workflow Completed for: '{original_user_question}' ---")


if __name__ == "__main__":
    async def run_all_examples():
        print("IntraIntel.ai Coding Challenge - Multi-LLM Agent (Separate Source Answers with Links)")
        print("="*70)
        
        medical_questions = [
            "What are the common treatments for type 2 diabetes?",
            "How effective is cognitive behavioral therapy for anxiety disorders?",
            "What are the latest research findings on long COVID symptoms?",
            "Can you explain the mechanism of action for mRNA vaccines?",
            "What are the risk factors and prevention strategies for osteoporosis?"
        ]
        refine_queries = True
        summarize_snippets = True

        for i, question in enumerate(medical_questions):
            await main_agent_workflow(
                question, 
                use_pubmed_for_this_question=True, 
                enable_query_refinement=refine_queries, 
                enable_snippet_summarization=summarize_snippets
            )
            if i < len(medical_questions) - 1: 
                pause_duration = 15 
                logger.info(f"Pausing for {pause_duration} seconds before next question...")
                await asyncio.sleep(pause_duration)
        print("\nAll medical questions processed.")
        print("="*70)
    asyncio.run(run_all_examples())