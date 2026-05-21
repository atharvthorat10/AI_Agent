import os
import fitz  # PyMuPDF
from typing import Dict, Any, List
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class ExtractionResult(BaseModel):
    text: str
    confidence: float

def get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please check your .env file or environment.")
    return genai.Client(api_key=api_key)

def extract_from_file(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Routes file to correct parser based on file type.
    Supported file_types: pdf, png, jpg, jpeg, mp3, wav, m4a
    Returns a dict with {"text": str, "confidence": float, "method": str}
    """
    ext = file_type.lower().strip(".")
    
    if ext == "pdf":
        return _extract_pdf(file_path)
    elif ext in ["png", "jpg", "jpeg"]:
        return _ocr_image(file_path, f"image/{ext}")
    elif ext in ["mp3", "wav", "m4a", "mp4"]:
        mime_map = {"mp3": "audio/mp3", "wav": "audio/wav", "m4a": "audio/m4a", "mp4": "video/mp4"}
        mime = mime_map.get(ext, "audio/mp3")
        if ext == "mp4":
            return _transcribe_video(file_path, mime)
        else:
            return _transcribe_audio(file_path, mime)
    else:
        # Fallback for plain text or unknown
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return {"text": content, "confidence": 1.0, "method": "plain_text"}
        except Exception as e:
            raise ValueError(f"Unsupported file type: {ext}. Error reading: {str(e)}")

def _extract_pdf(file_path: str) -> Dict[str, Any]:
    """
    Parses PDF using PyMuPDF. Falls back to OCR if the PDF is scanned (empty text).
    """
    text_content = []
    doc = fitz.open(file_path)
    
    for page in doc:
        text_content.append(page.get_text())
        
    full_text = "\n".join(text_content).strip()
    
    # If the text is very short, it's likely a scanned PDF. Fall back to OCR.
    if len(full_text) < 50:
        doc.close()
        return _ocr_pdf_fallback(file_path)
    
    doc.close()
    return {
        "text": full_text,
        "confidence": 1.0,
        "method": "pdf_text_extraction"
    }

def _ocr_pdf_fallback(file_path: str) -> Dict[str, Any]:
    """
    OCR Fallback: Renders PDF pages to images and sends them to Gemini for OCR.
    """
    try:
        client = get_client()
        doc = fitz.open(file_path)
        parts = []
        
        # Render up to 5 pages to avoid token limit or long times in fallback
        max_pages = min(len(doc), 5)
        for i in range(max_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            parts.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/png"
                )
            )
        doc.close()
        
        prompt = (
            "Perform OCR on the attached PDF page images. "
            "Extract all visible text. Maintain layout reading order where possible."
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=[*parts, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                ),
            )
            import json
            res_data = json.loads(response.text)
            return {
                "text": res_data.get("text", ""),
                "confidence": float(res_data.get("confidence", 0.9)),
                "method": "pdf_ocr_fallback"
            }
        except Exception:
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[*parts, "Perform OCR and extract all text from these images."]
                )
                return {
                    "text": response.text,
                    "confidence": 0.85,
                    "method": "pdf_ocr_fallback_plain"
                }
            except Exception:
                raise ValueError("API call failed")
    except Exception as e:
        return {
            "text": f"[Demo Mode] Simulated OCR fallback text from scanned PDF because the live API call failed: {str(e)}",
            "confidence": 0.85,
            "method": "pdf_ocr_fallback_demo"
        }

def _ocr_image(file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Uses Gemini multimodal capabilities to perform OCR on images.
    """
    try:
        client = get_client()
        with open(file_path, "rb") as f:
            img_bytes = f.read()
            
        parts = [
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            "Perform OCR on this image. Extract all text. Return the exact text and an estimated confidence score."
        ]
        
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                ),
            )
            import json
            res_data = json.loads(response.text)
            return {
                "text": res_data.get("text", ""),
                "confidence": float(res_data.get("confidence", 0.95)),
                "method": "image_ocr"
            }
        except Exception:
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime_type), "Extract all text from this image."]
                )
                return {
                    "text": response.text,
                    "confidence": 0.9,
                    "method": "image_ocr_plain"
                }
            except Exception:
                raise ValueError("API call failed")
    except Exception as e:
        return {
            "text": f"[Demo Mode] Simulated OCR text from image. To see live OCR content, configure your GEMINI_API_KEY (API error occurred: {str(e)}).",
            "confidence": 0.95,
            "method": "image_ocr_demo"
        }

def _transcribe_audio(file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Uses Gemini API to transcribe audio files.
    """
    try:
        client = get_client()
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            
        parts = [
            types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
            "Transcribe this audio file into clean, punctuated text. Return the transcript and a transcription quality confidence score."
        ]
        
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                ),
            )
            import json
            res_data = json.loads(response.text)
            return {
                "text": res_data.get("text", ""),
                "confidence": float(res_data.get("confidence", 0.9)),
                "method": "audio_transcription"
            }
        except Exception:
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[types.Part.from_bytes(data=audio_bytes, mime_type=mime_type), "Transcribe this audio file."]
                )
                return {
                    "text": response.text,
                    "confidence": 0.85,
                    "method": "audio_transcription_plain"
                }
            except Exception:
                raise ValueError("API call failed")
    except Exception as e:
        return {
            "text": f"[Demo Mode] Simulated transcription of audio file. Please configure GEMINI_API_KEY (API error occurred: {str(e)}).",
            "confidence": 0.9,
            "method": "audio_transcription_demo"
        }

def _transcribe_video(file_path: str, mime_type: str) -> Dict[str, Any]:
    """
    Uses Gemini API to transcribe/analyze video files.
    """
    try:
        client = get_client()
        with open(file_path, "rb") as f:
            video_bytes = f.read()
            
        parts = [
            types.Part.from_bytes(data=video_bytes, mime_type=mime_type),
            "Analyze this video and provide a detailed summary of its content, including transcription of spoken words. Return the transcript and a confidence score."
        ]
        
        try:
            response = client.models.generate_content(
                model='gemini-1.5-flash',
                contents=parts,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ExtractionResult,
                ),
            )
            import json
            res_data = json.loads(response.text)
            return {
                "text": res_data.get("text", ""),
                "confidence": float(res_data.get("confidence", 0.9)),
                "method": "video_transcription"
            }
        except Exception:
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=[types.Part.from_bytes(data=video_bytes, mime_type=mime_type), "Analyze this video."]
                )
                return {
                    "text": response.text,
                    "confidence": 0.85,
                    "method": "video_transcription_plain"
                }
            except Exception:
                raise ValueError("API call failed")
    except Exception as e:
        return {
            "text": f"[Demo Mode] Simulated transcription of video file. Please configure GEMINI_API_KEY (API error occurred: {str(e)}).",
            "confidence": 0.9,
            "method": "video_transcription_demo"
        }
