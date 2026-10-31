// Voice mode: stream mic audio to /ws/voice, show live transcripts, play the spoken reply.
// Text events (stage/token/done/blocked/error) are handed to the same handler as typed chat,
// so voice turns render exactly like text turns.
const micBtn = document.getElementById("mic-btn");
const liveLine = document.getElementById("live");
const voiceStatus = document.getElementById("voice-status");

let vws = null, micCtx = null, micStream = null, micNode = null;
let playCtx = null, nextStart = 0, activeSources = 0;
let voiceOn = false, micMuted = true, turnActive = false, ttsFinished = false;

function setVoiceStatus(text) { voiceStatus.textContent = text; }

async function startVoice() {
  if (voiceOn || !sessionId) return;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    addBubble("error", "Microphone unavailable: " + e.message + " (use http://localhost or HTTPS, and allow the microphone)");
    return;
  }
  micCtx = new AudioContext();
  playCtx = micCtx;                      // one context for capture and playback
  await micCtx.resume();
  await micCtx.audioWorklet.addModule("mic-worklet.js");
  const src = micCtx.createMediaStreamSource(micStream);
  micNode = new AudioWorkletNode(micCtx, "downsampler");
  const mute = micCtx.createGain();      // worklet must be connected to run; keep it silent
  mute.gain.value = 0;
  src.connect(micNode).connect(mute).connect(micCtx.destination);
  micNode.port.onmessage = (e) => {
    if (!micMuted && vws && vws.readyState === WebSocket.OPEN) vws.send(e.data);
  };

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  vws = new WebSocket(`${protocol}://${window.location.host}/ws/voice`);
  vws.binaryType = "arraybuffer";
  vws.onopen = () => vws.send(JSON.stringify({ type: "start", session_id: sessionId }));
  vws.onclose = () => { if (voiceOn) stopVoice(); };
  vws.onmessage = onVoiceMessage;
  voiceOn = true;
  micBtn.textContent = "Stop";
  micBtn.setAttribute("aria-pressed", "true");
  micBtn.classList.add("mic-on");
  setVoiceStatus("starting…");
}

function stopVoice() {
  voiceOn = false; micMuted = true; turnActive = false;
  if (vws) { vws.onclose = null; vws.close(); vws = null; }
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (micCtx) micCtx.close();
  micCtx = playCtx = micNode = micStream = null;
  nextStart = 0; activeSources = 0;
  liveLine.hidden = true;
  micBtn.textContent = "Speak";
  micBtn.setAttribute("aria-pressed", "false");
  micBtn.classList.remove("mic-on");
  setVoiceStatus("");
  setSending(false);
}

function onVoiceMessage(event) {
  if (event.data instanceof ArrayBuffer) return playChunk(event.data);
  const msg = JSON.parse(event.data);
  switch (msg.type) {
    case "ready":
      if (msg.greeting) {                // a spoken welcome-back follows; keep the mic closed until it ends
        micMuted = true; turnActive = true; ttsFinished = false;
        setVoiceStatus("speaking…");
      } else {
        micMuted = false;
        setVoiceStatus("listening…");
      }
      break;
    case "partial":
      liveLine.textContent = msg.text;
      liveLine.hidden = false;
      break;
    case "final":
      micMuted = true; turnActive = true; ttsFinished = false;
      liveLine.hidden = true;
      addBubble("user", msg.text);
      setSending(true);
      setVoiceStatus("thinking…");
      break;
    case "tts_done":
      ttsFinished = true;
      maybeResume();
      break;
    case "voice_metrics":
      addVoiceMetrics(msg);
      break;
    default:
      handleServerEvent(msg);            // stage | token | done | blocked | error (same as text chat)
  }
}

function playChunk(buf) {
  if (!playCtx) return;
  const rate = new DataView(buf).getUint32(4, true);
  const pcm = new Int16Array(buf, 8);    // offset 8 keeps 2-byte alignment
  const audio = playCtx.createBuffer(1, pcm.length, rate);
  const out = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) out[i] = pcm[i] / 32768;
  const node = playCtx.createBufferSource();
  node.buffer = audio;
  node.connect(playCtx.destination);
  const at = Math.max(playCtx.currentTime + 0.02, nextStart);   // queue back-to-back, in order
  node.start(at);
  nextStart = at + audio.duration;
  activeSources++;
  setVoiceStatus("speaking…");
  node.onended = () => { activeSources--; maybeResume(); };
}

function maybeResume() {
  if (!turnActive || !ttsFinished || activeSources > 0) return;
  turnActive = false;
  nextStart = 0;
  if (vws && vws.readyState === WebSocket.OPEN) vws.send(JSON.stringify({ type: "resume" }));
  micMuted = false;                      // only now do we listen again, so the bot never hears itself
  setVoiceStatus("listening…");
}

function addVoiceMetrics(m) {
  const f = (k, label) => (m[k] != null ? `${label} ${Math.round(m[k])} ms` : null);
  const parts = [f("asr_ms", "speech recognition"), f("pre_generation_ms", "retrieval/tools"),
                 f("llm_ttft_ms", "model first token"), f("tts_first_audio_ms", "speech synthesis"),
                 f("first_audio_after_speech_end_ms", "end of speech to first audio")].filter(Boolean);
  const div = document.createElement("div");
  div.className = "meta voice-meta";
  div.textContent = parts.join(", ");
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

micBtn.addEventListener("click", () => (voiceOn ? stopVoice() : startVoice()));
newChatBtn.addEventListener("click", () => { if (voiceOn) stopVoice(); });
