# main.py
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from dotenv import load_dotenv
load_dotenv()
import traceback
import tempfile
import shutil
import base64
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ws_audio import AudioSession, transcribe_pcm
from facial_session import FacialSession
from facial_module import analyze_frame, composite_score, label_from_score
from interview_bot import llm_judge, generate_followup, classify_answer
from video_evaluate import evaluate_uploaded_video, evaluate_video_url

from interview_bot import (
    llm_judge_no_reference, generate_followup, classify_answer,
    classify_utterance
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://interview-analyzer-2-j8oc.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
sessions = {}  # session_id -> {"audio": AudioSession, "facial": FacialSession, "current_q": {...}}


def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "audio": AudioSession(),
            "facial": FacialSession(),
            "current_q": None,
        }
    return sessions[session_id]


@app.websocket("/ws/video/{session_id}")
async def video_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sess = get_session(session_id)
    try:
        while True:
            data = await websocket.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            analysis = analyze_frame(frame)

            if analysis.get("face_detected"):
                blink_rate = sess["facial"].blink_rate_per_min()
                result = composite_score(analysis, blink_rate)
                sess["facial"].update(result["score"], analysis["eye_aspect_ratio"])
                await websocket.send_json({
                    "face_detected": True,
                    "score": result["score"],
                    "label": result["label"],
                })
            else:
                await websocket.send_json({
                    "face_detected": False,
                })
    except WebSocketDisconnect:
        pass
    
def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "audio": AudioSession(),
            "facial": FacialSession(),
            "current_question": None,  
            "conversation_history": [], # auto-detected, no longer set manually
        }
    return sessions[session_id]


@app.websocket("/ws/audio/{session_id}")
async def audio_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    sess = get_session(session_id)
    audio_sess = sess["audio"]
    try:
        while True:
            chunk = await websocket.receive_bytes()
            utterance = audio_sess.feed(chunk)

            if not utterance:
                continue

            transcript = transcribe_pcm(utterance)
            if not transcript.strip():
                continue

            kind = classify_utterance(transcript)

            if kind == "question":
                sess["current_question"] = transcript
                await websocket.send_json({
                    "type": "question_detected",
                    "question": transcript,
                })
                continue

            # It's an answer — only evaluate if we have a question to evaluate against
            if not sess["current_question"]:
                continue

            question = sess["current_question"]
            judged = llm_judge_no_reference(question, transcript, conversation_history=sess["conversation_history"])
            facial_score = sess["facial"].current_score()
            facial_label = label_from_score(facial_score)

            overall_confidence = round(
                0.7 * judged["confidence_in_answer"] +
                0.3 * facial_score, 1
            )

            classification = classify_answer(overall_confidence, correctness=judged.get("correctness"))
            
            response = {
                "type": "evaluation",
                "question": question,
                "facial_expression": facial_label,
                "facial_score": round(facial_score, 1),
                "transcript": transcript,
                "verdict": classification["verdict"],
                "reasoning": judged["reasoning"],
                "overall_confidence": overall_confidence,
                "followup_question": None,
            }

            if classification["needs_followup"]:
                response["followup_question"] = generate_followup(
                    question, transcript, conversation_history=sess["conversation_history"]
                )

            await websocket.send_json(response)
            sess["conversation_history"].append({"question": question, "answer": transcript})

            
    except WebSocketDisconnect:
        pass

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            "audio": AudioSession(),
            "facial": FacialSession(),
            "current_question": None,
            "conversation_history": [],   # <-- नया
        }
    return sessions[session_id]

@app.post("/session/{session_id}/set-question")
async def set_question(session_id: str, question: str, ideal_answer: str):
    sess = get_session(session_id)
    sess["current_q"] = {"question": question, "ideal_answer": ideal_answer}
    return {"status": "ok"}


@app.post("/evaluate-video")
async def evaluate_video_endpoint(video: UploadFile = File(...)):
    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name
    try:
        results = evaluate_uploaded_video(tmp_path)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        os.remove(tmp_path)
    return {"results": results}


@app.post("/evaluate-video-url")
async def evaluate_video_url_endpoint(video_url: str = Form(...)):
    try:
        results = evaluate_video_url(video_url)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    return {"results": results}