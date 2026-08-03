# video_evaluate.

import subprocess
import tempfile
import os
import glob
import cv2
import yt_dlp
from interview_bot import llm_judge, generate_followup, classify_answer
from facial_module import analyze_frame, composite_score, label_from_score
from facial_module import analyze_frame, composite_score
from ws_audio import stt_model
# from interview_bot import llm_judge, generate_followup
from interview_bot import llm_judge_no_reference, generate_followup, classify_answer, parse_qa_pairs_from_transcript
from facial_module import analyze_frame, composite_score, label_from_score

def download_video_from_url(url: str, out_dir: str) -> str:
    outtmpl = os.path.join(out_dir, "downloaded.%(ext)s")
    
    cookie_path = "/etc/secrets/cookies.txt" if os.path.exists("/etc/secrets/cookies.txt") else "cookies.txt"
    print(f"DEBUG: Using cookiefile = {cookie_path}, exists = {os.path.exists(cookie_path)}")
    
    ydl_opts = {
        "outtmpl": outtmpl,
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "max_filesize": 200 * 1024 * 1024,
        "cookiefile": cookie_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    ...
    mp4_candidates = glob.glob(os.path.join(out_dir, "downloaded.mp4"))
    if mp4_candidates:
        return mp4_candidates[0]

    all_candidates = glob.glob(os.path.join(out_dir, "downloaded.*"))
    print("WARNING: no merged .mp4 found, candidates were:", all_candidates)
    if not all_candidates:
        raise RuntimeError("yt-dlp did not produce any output file — download may have failed silently.")
    return all_candidates[0]

def extract_audio(video_path: str, out_wav_path: str) -> bool:
    """Returns True if audio was extracted, False if the video has no audio track."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn",
            out_wav_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "does not contain any stream" in result.stderr or "Output file does not contain any stream" in result.stderr:
            print("WARNING: video has no audio track — proceeding with empty transcript.")
            return False
        print("FFMPEG STDERR (extract_audio):", result.stderr)
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")
    return True

    if result.returncode != 0:
        print("FFMPEG STDERR (extract_audio):", result.stderr)
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-500:]}")


def extract_frames(video_path: str, out_dir: str, fps: int = 1):
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", f"fps={fps}",
            os.path.join(out_dir, "frame_%04d.jpg"),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "does not contain any stream" in result.stderr or "Output file does not contain any stream" in result.stderr:
            print("WARNING: video has no video track — proceeding with no facial data.")
            return False
        print("FFMPEG STDERR (extract_frames):", result.stderr)
        raise RuntimeError(f"ffmpeg frame extraction failed: {result.stderr[-500:]}")
    return True

def _run_pipeline(video_path: str) -> list:
    """Returns a list of evaluated Q&A results — question and ideal_answer are no longer inputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = os.path.join(tmp_dir, "audio.wav")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        has_audio = extract_audio(video_path, wav_path)
        if has_audio:
            segments, _ = stt_model.transcribe(
            wav_path,
            beam_size=10,
            # language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
)
            transcript = " ".join(seg.text for seg in segments).strip()
        else:
            transcript = ""

        has_video = extract_frames(video_path, frames_dir, fps=1)
        if has_video:
            frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        else:
            frame_paths = []
            
        facial_scores = []
        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is None:
                continue
            analysis = analyze_frame(frame)
            if analysis.get("face_detected"):
                result = composite_score(analysis, blink_rate_per_min=0)
                facial_scores.append(result["score"])

        avg_facial_score = sum(facial_scores) / len(facial_scores) if facial_scores else 50.0
        avg_facial_label = label_from_score(avg_facial_score)

        if not transcript:
            return [{
                "question": None,
                "transcript": "",
                "facial_expression": avg_facial_label,
                "facial_score": round(avg_facial_score, 1),
                "verdict": "incorrect",
                "reasoning": "No audio/speech detected in this video.",
                "overall_confidence": 0,
                "followup_question": None,
            }]

        print("=== RAW TRANSCRIPT ===")
        print(transcript)
        print("=== END TRANSCRIPT ===")

        qa_pairs = parse_qa_pairs_from_transcript(transcript)

        print("=== PARSED QA PAIRS ===")
        print(qa_pairs)
        print("=== END QA PAIRS ===")

        qa_pairs = parse_qa_pairs_from_transcript(transcript)

        if not qa_pairs:
            return [{
                "question": None,
                "transcript": transcript,
                "facial_expression": avg_facial_label,
                "facial_score": round(avg_facial_score, 1),
                "verdict": "incorrect",
                "reasoning": "No clear question-answer exchange was detected in this transcript.",
                "overall_confidence": 0,
                "followup_question": None,
            }]

        results = []

        qa_pairs = parse_qa_pairs_from_transcript(transcript)
        # ensure correct order even if the model didn't sort them
        qa_pairs = sorted(qa_pairs, key=lambda p: p.get("turn_index", 0))

        results = []
        conversation_history = []

        for pair in qa_pairs:
            question = pair.get("question", "")
            answer = pair.get("answer", "")

            judged = llm_judge_no_reference(question, answer, conversation_history=conversation_history)
            overall_confidence = round(
                0.7 * judged["confidence_in_answer"] +
                0.3 * avg_facial_score, 1
            )
            classification = classify_answer(overall_confidence, correctness=judged.get("correctness"))

            result = {
                "question": question,
                "facial_expression": avg_facial_label,
                "facial_score": round(avg_facial_score, 1),
                "transcript": answer,
                "verdict": classification["verdict"],
                "reasoning": judged["reasoning"],
                "overall_confidence": overall_confidence,
                "followup_question": None,
            }

            if classification["needs_followup"]:
                result["followup_question"] = generate_followup(
                    question, answer, conversation_history=conversation_history
                )

            results.append(result)
            conversation_history.append({"question": question, "answer": answer})  # <-- context carries forward

        return results

def evaluate_uploaded_video(video_path: str) -> list:
    return _run_pipeline(video_path)


def evaluate_video_url(url: str) -> list:
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded_path = download_video_from_url(url, tmp_dir)
        return _run_pipeline(downloaded_path)