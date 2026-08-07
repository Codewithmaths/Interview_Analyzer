// interview_client.js

const $ = (id) => document.getElementById(id);

let audioWs = null;
let videoWs = null;
let mediaStream = null;
let audioCtx = null;
let processorNode = null;
let videoInterval = null;

function wsBase() {
   const host = $("serverUrl").value.trim();
   const isLocal = host.startsWith("localhost") || host.startsWith("127.0.0.1");
   return (isLocal ? "ws://" : "wss://") + host;
}

function httpBase() {
   const host = $("serverUrl").value.trim();
   const isLocal = host.startsWith("localhost") || host.startsWith("127.0.0.1");
   return (isLocal ? "http://" : "https://") + host;
}

function setDot(id, state) {
   const el = $(id);
   el.classList.remove("connected", "error");
   if (state === "connected") el.classList.add("connected");
   if (state === "error") el.classList.add("error");
}

// --- Toggle between upload / url blocks ---
document.querySelectorAll('input[name="videoSource"]').forEach((radio) => {
   radio.addEventListener("change", (e) => {
      $("uploadBlock").style.display = e.target.value === "upload" ? "block" : "none";
      $("urlBlock").style.display = e.target.value === "url" ? "block" : "none";
   });
});

// --- Start live interview ---
$("startBtn").addEventListener("click", async () => {
   const sessionId = $("sessionId").value.trim();

   try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
   } catch (err) {
      alert("Could not access camera/mic: " + err.message);
      return;
   }

   $("preview").srcObject = mediaStream;

   audioWs = new WebSocket(`${wsBase()}/ws/audio/${sessionId}`);
   audioWs.binaryType = "arraybuffer";
   audioWs.onopen = () => setDot("audioDot", "connected");
   audioWs.onclose = () => setDot("audioDot", "");
   audioWs.onerror = () => setDot("audioDot", "error");
   audioWs.onmessage = (event) => {
      try {
         const result = JSON.parse(event.data);
         if (result.type === "question_detected") {
            renderQuestionDetected(result.question);
         } else if (result.type === "evaluation") {
            renderEvaluation(result);
         }
      } catch (e) {
         console.warn("Non-JSON message from audio WS:", event.data);
      }
   };

   videoWs = new WebSocket(`${wsBase()}/ws/video/${sessionId}`);
   videoWs.binaryType = "arraybuffer";
   videoWs.onopen = () => setDot("videoDot", "connected");
   videoWs.onclose = () => setDot("videoDot", "");
   videoWs.onerror = () => setDot("videoDot", "error");
   videoWs.onmessage = (event) => {
      try {
         const result = JSON.parse(event.data);
         if (result.face_detected === false) {
            $("facialLabel").textContent = "Face Not Detected";
         } else if (result.label) {
            $("facialLabel").textContent = result.label;
         }
      } catch (e) {
         // ignore
      }
   };

   startAudioStream();
   startVideoStream();

   $("startBtn").disabled = true;
   $("stopBtn").disabled = false;
});

$("stopBtn").addEventListener("click", stopAll);

function stopAll() {
   if (processorNode) { processorNode.disconnect(); processorNode = null; }
   if (audioCtx) { audioCtx.close(); audioCtx = null; }
   if (videoInterval) { clearInterval(videoInterval); videoInterval = null; }
   if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
   if (audioWs) { audioWs.close(); audioWs = null; }
   if (videoWs) { videoWs.close(); videoWs = null; }

   setDot("audioDot", "");
   setDot("videoDot", "");
   $("startBtn").disabled = false;
   $("stopBtn").disabled = true;

   $("facialLabel").textContent = "no data";
}

// --- Continuous audio streaming (16kHz PCM16) ---
function startAudioStream() {
   audioCtx = new AudioContext({ sampleRate: 16000 });
   const source = audioCtx.createMediaStreamSource(mediaStream);
   processorNode = audioCtx.createScriptProcessor(4096, 1, 1);

   source.connect(processorNode);
   processorNode.connect(audioCtx.destination);

   processorNode.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
         const s = Math.max(-1, Math.min(1, input[i]));
         pcm16[i] = s < 0 ? s * 32768 : s * 32767;
      }
      if (audioWs && audioWs.readyState === WebSocket.OPEN) {
         audioWs.send(pcm16.buffer);
      }
   };
}

// --- Webcam frame streaming (~1 fps) ---
function startVideoStream() {
   const canvas = document.createElement("canvas");
   canvas.width = 320;
   canvas.height = 240;
   const ctx = canvas.getContext("2d");
   const video = $("preview");

   videoInterval = setInterval(() => {
      if (!video.videoWidth) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
         if (blob && videoWs && videoWs.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => videoWs.send(buf));
         }
      }, "image/jpeg", 0.7);
   }, 1000);
}

// --- Evaluate offline video (upload OR url) — no question/ideal-answer needed ---
$("evaluateVideoBtn").addEventListener("click", async () => {
   const statusEl = $("videoEvalStatus");
   const source = document.querySelector('input[name="videoSource"]:checked').value;

   statusEl.textContent = "Processing... (may take 10-60s depending on length)";

   try {
      let res;
      if (source === "upload") {
         const fileInput = $("videoFile");
         if (!fileInput.files.length) {
            statusEl.textContent = "Choose a video file first.";
            return;
         }
         const formData = new FormData();
         formData.append("video", fileInput.files[0]);
         res = await fetch(`${httpBase()}/evaluate-video`, { method: "POST", body: formData });
      } else {
         const videoUrl = $("videoUrl").value.trim();
         if (!videoUrl) {
            statusEl.textContent = "Enter a video URL first.";
            return;
         }
         const formData = new FormData();
         formData.append("video_url", videoUrl);
         res = await fetch(`${httpBase()}/evaluate-video-url`, { method: "POST", body: formData });
      }

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (data.error) {
         statusEl.textContent = `Failed: ${data.error}`;
         return;
      }

      if (Array.isArray(data.results)) {
         data.results.forEach(renderEvaluation);
      } else {
         // fallback in case a single-object shape ever comes back
         renderEvaluation(data);
      }
      statusEl.textContent = "Done ✓";
   } catch (err) {
      statusEl.textContent = `Failed: ${err.message}`;
   }
});

function renderQuestionDetected(question) {
   const log = $("log");
   if (log.querySelector(".hint")) log.innerHTML = "";
   const entry = document.createElement("div");
   entry.className = "entry";
   entry.innerHTML = `<div class="question-detected">🎤 Question detected: <strong>${escapeHtml(question)}</strong></div>`;
   log.appendChild(entry);
   log.scrollTop = log.scrollHeight;
}

function renderEvaluation(result) {
   const log = $("log");
   if (log.querySelector(".hint")) log.innerHTML = "";

   const entry = document.createElement("div");
   entry.className = "entry";

   const verdictClassMap = {
      correct: "correct",
      incorrect: "incorrect",
      partially_correct: "partially_correct",
      not_confirmed: "not_confirmed",
   };

   const verdictLabelMap = {
      correct: "Correct",
      incorrect: "Incorrect",
      partially_correct: "Partially Correct",
      not_confirmed: "Not Confirmed",
   };
   
   const badgeClass = verdictClassMap[result.verdict] || "partial_correct";
   const verdictLabel = verdictLabelMap[result.verdict] || (result.verdict || "unknown");

   entry.innerHTML = `
    ${result.question ? `<div class="meta">Q: <strong>${escapeHtml(result.question)}</strong></div>` : ""}
    <div class="meta">Facial expression: <strong>${escapeHtml(result.facial_expression || "—")}</strong> (score: ${result.facial_score ?? "—"})</div>
    <span class="badge ${badgeClass}" style="margin-top:6px; display:inline-block;">${verdictLabel}</span>
    <span class="meta">confidence: ${result.overall_confidence ?? "—"}</span>
    <div class="transcript">"${escapeHtml(result.transcript || "")}"</div>
    <div class="meta">${escapeHtml(result.reasoning || "")}</div>
    ${result.followup_question ? `<div class="followup">Follow-up: ${escapeHtml(result.followup_question)}</div>` : ""}
  `;

   log.appendChild(entry);
   log.scrollTop = log.scrollHeight;
}

function escapeHtml(str) {
   const div = document.createElement("div");
   div.textContent = str;
   return div.innerHTML;
}