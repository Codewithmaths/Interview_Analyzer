# interview_bot.py
from dotenv import load_dotenv
load_dotenv()

import os
import json
from sentence_transformers import SentenceTransformer, util
from groq import Groq

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Used for optional reference-answer comparison (kept available, no longer required by default flow)
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

GROQ_MODEL = "llama-3.3-70b-versatile"

def semantic_score(candidate_answer: str, ideal_answer: str) -> float:
    """Cosine-similarity based score (0-100) between a candidate answer and a reference answer.
    Only used if you still want reference-answer comparison somewhere."""
    if not candidate_answer or not ideal_answer:
        return 0.0
    emb1 = similarity_model.encode(candidate_answer, convert_to_tensor=True)
    emb2 = similarity_model.encode(ideal_answer, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2).item()
    return max(0, min(1, sim)) * 100


def _clean_json_response(content: str) -> str:
    """Strips markdown code fences some LLM responses wrap JSON in."""
    content = content.strip()
    if content.startswith("```json"):
        content = content[len("```json"):]
    if content.startswith("```"):
        content = content[len("```"):]
    if content.endswith("```"):
        content = content[:-len("```")]
    return content.strip()


def llm_judge(question: str, candidate_answer: str, ideal_answer: str) -> dict:
    """Judges an answer against a provided reference/ideal answer."""
    prompt = f"""
You are an interview evaluator. Given the question, the ideal answer, and the candidate's answer,
return STRICT JSON only (no markdown, no preamble) with keys:
- "correctness": "correct" | "incorrect" | "partial"
- "reasoning": short 1-line reason
- "confidence_in_answer": integer 0-100 (how confidently/clearly the answer addresses the question)

Question: {question}
Ideal Answer: {ideal_answer}
Candidate Answer: {candidate_answer}
"""
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = _clean_json_response(resp.choices[0].message.content)
    return json.loads(content)


def llm_judge_no_reference(question: str, answer: str, conversation_history: list = None) -> dict:
    history_text = ""
    if conversation_history:
        history_lines = [
            f"Turn {i+1} — Q: {t['question']} | A: {t['answer']}"
            for i, t in enumerate(conversation_history)
        ]
        history_text = "Prior turns in this same interview (for context only):\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""
You are an interview evaluator. Judge the candidate's CURRENT answer to the question below.

IMPORTANT: The current question or answer may contain pronouns (she, he, it, they, that person,
etc.) that refer back to a person, place, or topic mentioned in the PRIOR TURNS above. You MUST
resolve these pronouns using the prior turns before judging — do not treat an answer as vague or
unclear just because it uses a pronoun, if the prior turns make the reference obvious. Only flag
an answer as "not_confirmable" or unclear if the reference genuinely cannot be resolved even using
the full prior context.

Classify "correctness" as exactly one of these four values:
- "correct" — the answer is verifiably right
- "incorrect" — the answer is verifiably wrong
- "partial" — the answer is incomplete or only partly addresses the question
- "not_confirmable" — the question asks for a personal fact/claim that cannot be verified as
  true or false (not the same as "ambiguous due to an unresolved pronoun" — resolve pronouns first)

If the question involves any calculation, counting, or objectively verifiable fact, work it out
yourself step by step first, then compare to the candidate's answer exactly.

{"Use the prior turns to resolve any pronouns or references in the current question/answer, and for general context on what stage of the conversation this is — do not judge the prior turns themselves." if conversation_history else ""}

Return STRICT JSON only (no markdown, no preamble) with keys:
- "correctness": "correct" | "incorrect" | "partial" | "not_confirmable"
- "reasoning": short 1-line reason, factually consistent with the candidate's actual answer text
- "confidence_in_answer": integer 0-100

{history_text}Current Question: {question}
Current Candidate Answer: {answer}
"""
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = _clean_json_response(resp.choices[0].message.content)
    result = json.loads(content)

    try:
        result["confidence_in_answer"] = float(result.get("confidence_in_answer", 0))
    except (TypeError, ValueError):
        result["confidence_in_answer"] = 0

    return result

def classify_utterance(text: str) -> str:
    """Classifies a transcribed utterance as 'question' or 'answer'."""
    prompt = f"""Classify this spoken interview utterance as exactly one word: "question" or "answer".
A question is something an interviewer would ask a candidate. An answer is a response a candidate gives.
Return only the single word, nothing else, no punctuation.

Utterance: {text}"""
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    label = resp.choices[0].message.content.strip().lower()
    return "question" if "question" in label else "answer"

def parse_qa_pairs_from_transcript(transcript: str) -> list:
    prompt = f"""
The following is a transcript of a spoken interview/discussion, with no punctuation cues for who
is speaking, and sometimes with little to no punctuation at all marking sentence boundaries.
Reconstruct the actual substantive question-answer turns.

IMPORTANT RULES:
1. Only treat something as a "question" if it is a genuine substantive question requiring a real
   answer (e.g., asking for a fact, explanation, opinion, or calculation).
2. Do NOT treat the following as questions, even if phrased with a question mark or rising
   intonation: filler confirmations ("Okay?", "Right?", "You say that...?"), the interviewer
   restating/paraphrasing what the candidate just said, or brief acknowledgments. Skip these
   entirely — do not create a turn for them.
3. Skip trivial filler as ANSWERS too ("Okay.", "Yes.", "Good, very good.", "That's a happy
   answer.") when it is merely a closing remark/compliment from the interviewer rather than the
   candidate's actual substantive answer — do not attach filler closing remarks to a question as
   if they were the answer. If the real substantive answer appears earlier in the same run-on
   text, use THAT as the answer instead of trailing filler.
4. Sometimes the interviewer asks a question, then immediately rephrases or clarifies it in a
   follow-up sentence BEFORE the candidate answers. Merge all such consecutive question sentences
   into ONE combined question.
5. When there is little punctuation, use meaning and natural speech patterns to infer where the
   question ends and the candidate's actual substantive answer begins — the answer is typically
   the part that directly responds to the question's content, not a closing remark like "is it",
   "yes sir", "okay", or a compliment that comes after.
6. Never produce a turn with an empty answer, and never use pure filler/compliment text as the
   answer if a real substantive answer exists earlier in the text.
7. Preserve original order.

Return STRICT JSON only (no markdown, no preamble) — an ordered list of objects, each with keys
"turn_index" (integer, starting at 1), "question", and "answer". If there are no genuine
substantive questions at all, return an empty list [].

Example: "before the mountain was discovered which was the tallest mountain sir even without
discovering also it would remain the tallest mountain only sir is it yes sir okay good very good"
Correct output: question = "before the mountain was discovered which was the tallest mountain",
answer = "even without discovering also it would remain the tallest mountain only sir". The
trailing "is it yes sir okay good very good" is filler/confirmation and should be discarded, not
used as part of the question or answer.

Now process this transcript:
{transcript}
"""
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = _clean_json_response(resp.choices[0].message.content)
    try:
        pairs = json.loads(content)
        return pairs if isinstance(pairs, list) else []
    except Exception:
        return []    
    
def classify_answer(overall_confidence: float, correctness: str = None) -> dict:
    """Maps LLM correctness + confidence score to one of 4 final verdicts:
    correct, incorrect, partially_correct, not_confirmed."""

    if correctness == "not_confirmable":
        return {"verdict": "not_confirmed", "needs_followup": False}

    if overall_confidence >= 75:
        verdict = "correct"
        needs_followup = False
    elif overall_confidence >= 40:
        verdict = "partially_correct"
        needs_followup = True
    else:
        verdict = "incorrect"
        needs_followup = False

    return {"verdict": verdict, "needs_followup": needs_followup}


def generate_followup(question: str, candidate_answer: str, conversation_history: list = None) -> str:
    history_text = ""
    if conversation_history:
        history_lines = [
            f"Turn {i+1} — Q: {t['question']} | A: {t['answer']}"
            for i, t in enumerate(conversation_history)
        ]
        history_text = "Prior turns in this same interview (use this to resolve any pronouns/context):\n" + "\n".join(history_lines) + "\n\n"

    prompt = f"""
The candidate gave a partially correct / unclear answer to this interview question.
{"Use the prior turns to understand any pronouns or references in the current question/answer before deciding what to ask." if conversation_history else ""}
Ask ONE short, targeted follow-up question that probes deeper into their answer
to clarify or test true understanding. Do NOT ask them to clarify a pronoun or reference that is
already clear from the prior turns. Return only the question text.

{history_text}Original Question: {question}
Candidate Answer: {candidate_answer}
"""
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()