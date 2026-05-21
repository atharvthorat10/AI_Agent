import os
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class PlannerDecision(BaseModel):
    is_ambiguous: bool = Field(description="True if the user's goal or task is not clear or if multiple tasks are equally plausible.")
    follow_up_question: Optional[str] = Field(None, description="Short, clear follow-up question if is_ambiguous is True. Otherwise null.")
    task_type: Optional[str] = Field(None, description="One of: image_pdf_ocr, youtube_transcript, conversation, summarize, sentiment, code_explain, audio_transcribe_summary")
    plan_steps: List[str] = Field(default_factory=list, description="List of sequential steps to execute this plan.")
    reasoning: str = Field(description="Brief explanation of the decision.")

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)

def estimate_cost(
    query: str, 
    extracted_text: Optional[str], 
    file_type: Optional[str], 
    task_type: Optional[str],
    audio_duration: float = 0.0
) -> float:
    """
    Estimates token and API cost for the proposed plan.
    Pricing for gemini-2.0-flash (approximate):
    - Input text: $0.075 / 1M tokens ($0.000000075 per token)
    - Output text: $0.30 / 1M tokens ($0.00000030 per token)
    - Image: 258 tokens per image
    - Audio: 32 tokens per second
    """
    # Estimate input tokens
    char_count = len(query) + (len(extracted_text) if extracted_text else 0)
    input_tokens = char_count // 4  # Rough character-to-token ratio (1 token ~ 4 chars)
    
    if file_type:
        ext = file_type.lower().strip(".")
        if ext in ["png", "jpg", "jpeg"]:
            input_tokens += 258
        elif ext in ["mp3", "wav", "m4a"]:
            # Default to 60s if duration not parsed yet
            duration = audio_duration if audio_duration > 0 else 60.0
            input_tokens += int(duration * 32)
            
    # Estimate output tokens based on task
    expected_output_tokens = 500
    if task_type == "code_explain":
        expected_output_tokens = 800
    elif task_type == "summarize":
        expected_output_tokens = 400
    elif task_type == "sentiment":
        expected_output_tokens = 150
        
    input_cost = input_tokens * 0.000000075
    output_cost = expected_output_tokens * 0.00000030
    
    total_cost = input_cost + output_cost
    return round(total_cost, 6)

def plan_task(
    query: str, 
    extracted_text: Optional[str] = None, 
    file_type: Optional[str] = None,
    audio_duration: float = 0.0
) -> Dict[str, Any]:
    """
    Analyzes inputs to decide if they are ambiguous or ready.
    If ready, generates a step-by-step execution plan and estimates costs.
    """
    try:
        client = get_client()
        
        context = ""
        if extracted_text:
            context += f"Uploaded file type: {file_type}\nExtracted text preview (first 1000 chars):\n{extracted_text[:1000]}\n\n"
            
        prompt = (
            f"You are an Agentic Planner. Review the user's query and the context of the uploaded file:\n\n"
            f"User Query: {query}\n"
            f"{context}"
            f"Determine if the user's intent is ambiguous. "
            f"If they uploaded a file and wrote a vague query (e.g. 'here', 'process', 'look at this') or wrote nothing at all, "
            f"you MUST mark is_ambiguous=true and provide a short, clear follow-up question asking for clarification. "
            f"If the goal is clear, classify it into one of these task_types:\n"
            f"- `image_pdf_ocr` (User wants text extraction only, or just uploaded document and said 'extract text')\n"
            f"- `youtube_transcript` (User wants to fetch/parse a YouTube video's captions from a URL)\n"
            f"- `conversation` (General chitchat, simple query, or QA on text without requiring summary/sentiment/code explanation)\n"
            f"- `summarize` (User explicitly asked to summarize the content)\n"
            f"- `sentiment` (User explicitly asked for sentiment analysis)\n"
            f"- `code_explain` (User provided code and asked to explain, debug, or analyze complexity)\n"
            f"- `audio_transcribe_summary` (User uploaded audio and wants transcription + summary)\n\n"
            f"Return a structured plan with steps. Be precise."
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PlannerDecision,
            ),
        )
        import json
        decision = json.loads(response.text)
        
        is_ambiguous = decision.get("is_ambiguous", True)
        follow_up = decision.get("follow_up_question")
        task_type = decision.get("task_type")
        plan_steps = decision.get("plan_steps", [])
        reasoning = decision.get("reasoning", "No reasoning provided.")
        
        # File dependency validation
        if task_type in ["image_pdf_ocr", "audio_transcribe_summary"] and not file_type:
            is_ambiguous = True
            follow_up = "You requested a file-specific task, but no file was uploaded. Please upload the file."
            task_type = None
            plan_steps = []
            reasoning = "Missing required file upload for the requested task."

        # Calculate cost
        cost = estimate_cost(query, extracted_text, file_type, task_type, audio_duration)
        
        return {
            "status": "ambiguous" if is_ambiguous else "ready",
            "follow_up_question": follow_up,
            "task_type": task_type,
            "plan": [{"step": idx + 1, "description": step} for idx, step in enumerate(plan_steps)],
            "reasoning": reasoning,
            "cost_estimate": cost
        }
        
    except Exception as e:
        # Fail-safe local heuristics
        query_lower = query.lower().strip()
        
        # 1. Check YouTube URL
        from app.agent.tasks import extract_youtube_video_id
        if extract_youtube_video_id(query):
            return {
                "status": "ready",
                "follow_up_question": None,
                "task_type": "youtube_transcript",
                "plan": [{"step": 1, "description": "Detect and extract YouTube video ID"}, {"step": 2, "description": "Fetch transcripts from API"}],
                "reasoning": "Detected YouTube URL in query. Routing to transcript fetcher.",
                "cost_estimate": 0.0001
            }
            
        # 2. Vague input fallback
        if not query_lower and extracted_text:
            return {
                "status": "ambiguous",
                "follow_up_question": "What would you like me to do with this file? (e.g. summarize it, analyze sentiment, or extract text?)",
                "task_type": None,
                "plan": [],
                "reasoning": "File uploaded but no query instructions provided.",
                "cost_estimate": 0.0
            }
            
        # 3. Local keywords matching
        summary_keywords = ["summarize", "summary", "summazrize", "sumerize", "summari", "brief", "outline", "condense", "gist"]
        sentiment_keywords = ["sentiment", "opinion", "tone", "emotion", "feeling", "positive", "negative", "neutral"]
        code_keywords = ["code", "bug", "complexity", "explain code", "debug", "python", "javascript", "c++", "java", "function", "class"]
        ocr_keywords = ["ocr", "extract text", "read text", "transcribe document"]
        audio_keywords = ["audio", "transcribe audio", "listen", "voice", "speech"]
        
        matched_task = None
        plan_steps = []
        reasoning_str = ""
        
        if any(kw in query_lower for kw in summary_keywords):
            matched_task = "summarize"
            plan_steps = [
                "Locate source text from file or query",
                "Format output into 1-line, 3-bullet points, and 5-sentence summaries"
            ]
            reasoning_str = f"Matched summary intent by keyword in query: '{query}'"
        elif any(kw in query_lower for kw in sentiment_keywords):
            matched_task = "sentiment"
            plan_steps = [
                "Locate source text for analysis",
                "Determine sentiment label (Positive/Negative/Neutral) and confidence"
            ]
            reasoning_str = f"Matched sentiment intent by keyword in query: '{query}'"
        elif any(kw in query_lower for kw in code_keywords):
            matched_task = "code_explain"
            plan_steps = [
                "Extract code snippet from query or document",
                "Analyze code logic, spot bugs, and compute time/space complexity"
            ]
            reasoning_str = f"Matched code explanation intent by keyword in query: '{query}'"
        elif any(kw in query_lower for kw in ocr_keywords):
            matched_task = "image_pdf_ocr"
            plan_steps = [
                "Run OCR extraction on image or scanned document page(s)",
                "Format and output plain text"
            ]
            reasoning_str = f"Matched OCR/text extraction intent by keyword in query: '{query}'"
        elif any(kw in query_lower for kw in audio_keywords) or (file_type and file_type.lower().strip(".") in ["mp3", "wav", "m4a"]):
            matched_task = "audio_transcribe_summary"
            plan_steps = [
                "Transcribe input audio file content",
                "Generate summary formatted output"
            ]
            reasoning_str = "Matched audio intent by keyword or file format."
            
        if matched_task:
            if matched_task in ["image_pdf_ocr", "audio_transcribe_summary"] and not file_type:
                return {
                    "status": "ambiguous",
                    "follow_up_question": "You requested a file-specific task, but no file was uploaded. Please upload the file.",
                    "task_type": None,
                    "plan": [],
                    "reasoning": "Missing required file upload for the requested task.",
                    "cost_estimate": 0.0
                }
            return {
                "status": "ready",
                "follow_up_question": None,
                "task_type": matched_task,
                "plan": [{"step": idx + 1, "description": step} for idx, step in enumerate(plan_steps)],
                "reasoning": f"Local intent classification: {reasoning_str} (fallback due to planning API error: {str(e)})",
                "cost_estimate": 0.0001
            }
            
        # Standard conversation fallback
        return {
            "status": "ready",
            "follow_up_question": None,
            "task_type": "conversation",
            "plan": [{"step": 1, "description": "Generate friendly conversational response"}],
            "reasoning": f"Defaulting to conversational response due to planning exception: {str(e)}",
            "cost_estimate": 0.0001
        }
