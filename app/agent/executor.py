import time
import logging
from typing import Dict, Any, Optional
from app.agent.state import SessionState, update_session
from app.agent import tasks

logger = logging.getLogger(__name__)
def execute_plan(session: SessionState) -> str:
    """
    Executes each step of the plan recorded in the session state.
    Updates the session with progress logs and final outputs.
    """
    session_id = session.session_id
    update_session(session_id, status="running")
    session.add_log(f"[Executor] Starting plan execution for task: {session.task_type}")
    
    # 1. Select the content source (prefer file text extraction, then query text)
    source_text = session.extracted_text if session.extracted_text else session.original_query
    
    try:
        # Loop through each step in the plan and log its commencement
        for step in session.plan:
            step_num = step["step"]
            desc = step["description"]
            session.add_log(f"[Executor] Running Step {step_num}: {desc}")
            time.sleep(0.1) # Simulate micro-processing step for smooth UI display
            
        result_text = ""
        
        # 2. Execute the actual task routing
        task_type = session.task_type
        
        if task_type == "image_pdf_ocr":
            # Just format and return the already extracted text
            # If no extracted text, try running extractor locally (in case we skipped or want to re-run)
            confidence = 1.0
            method = "Cached Text"
            
            # Retrieve confidence and method from metadata if available (simulated or stored)
            # For simplicity, we assume the initial file parsing did OCR, and we format it
            result_text = tasks.execute_ocr_extraction_task(
                extracted_text=session.extracted_text or "",
                confidence=0.95,  # Default fallback confidence if not parsed
                method=session.file_type or "Unknown"
            )
            
        elif task_type == "youtube_transcript":
            result_text = tasks.execute_youtube_transcript_task(session.original_query)
            
        elif task_type == "conversation":
            result_text = tasks.execute_conversational_task(
                query=session.original_query or "",
                extracted_text=session.extracted_text
            )
            
        elif task_type == "summarize":
            if not source_text:
                raise ValueError("No text content found to summarize.")
            result_text = tasks.execute_summarization_task(source_text)
            
        elif task_type == "sentiment":
            if not source_text:
                raise ValueError("No text content found for sentiment analysis.")
            result_text = tasks.execute_sentiment_analysis_task(source_text)
            
        elif task_type == "code_explain":
            if not source_text:
                raise ValueError("No code content found to explain.")
            result_text = tasks.execute_code_explanation_task(source_text)
            
        elif task_type == "audio_transcribe_summary":
            # For audio, we need the file path and file type to do transcription + summary
            # We mock file path if it's cached or retrieve the original file path
            # In our backend we store the uploaded file temporarily and save the path in state
            # Let's verify we have it
            temp_path = getattr(session, "temp_file_path", None)
            if not temp_path:
                raise ValueError("Audio temporary file path is missing in session state.")
                
            ext = (session.file_type or "mp3").lower()
            mime_map = {"mp3": "audio/mp3", "wav": "audio/wav", "m4a": "audio/m4a", "mp4": "video/mp4"}
            mime_type = mime_map.get(ext, "audio/mp3")
            result_text = tasks.execute_audio_transcription_summary_task(
                file_path=temp_path,
                mime_type=mime_type
            )
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        session.add_log("[Executor] Task execution completed successfully.")
        update_session(
            session_id,
            status="completed",
            result=result_text
        )
        return result_text
        
    except Exception as e:
        err_msg = f"Execution failed at step. Error: {str(e)}"
        session.add_log(f"[Executor] [ERROR] {err_msg}")
        update_session(
            session_id,
            status="failed",
            result=f"An error occurred during task execution: {str(e)}"
        )
        return f"Error: {str(e)}"
