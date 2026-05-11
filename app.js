import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const startBtn = document.getElementById("startBtn");

const EAR_THRESHOLD = 0.22;
const DROWSY_MS = 1500;

const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [263, 387, 385, 362, 380, 373];

let landmarker = null;
let closedSince = null;
let audioCtx = null;

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function ear(lm, idx) {
  const p = idx.map(i => lm[i]);
  return (dist(p[1], p[5]) + dist(p[2], p[4])) / (2 * dist(p[0], p[3]));
}

function beep() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.frequency.value = 880;
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.3);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.3);
}

let lastBeep = 0;
function alertDrowsy() {
  statusEl.textContent = "DROWSY! WAKE UP";
  statusEl.classList.add("drowsy");
  const now = performance.now();
  if (now - lastBeep > 800) {
    beep();
    lastBeep = now;
  }
}

function clearAlert() {
  statusEl.textContent = "Awake";
  statusEl.classList.remove("drowsy");
}

async function init() {
  startBtn.disabled = true;
  startBtn.textContent = "Loading...";

  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  landmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "./face_landmarker.task",
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1
  });

  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  startBtn.style.display = "none";
  loop();
}

function loop() {
  if (video.readyState >= 2) {
    const res = landmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (res.faceLandmarks && res.faceLandmarks.length > 0) {
      const lm = res.faceLandmarks[0];
      const leftEAR = ear(lm, LEFT_EYE);
      const rightEAR = ear(lm, RIGHT_EYE);
      const avg = (leftEAR + rightEAR) / 2;
      statsEl.textContent = `EAR: ${avg.toFixed(3)}`;

      ctx.strokeStyle = avg < EAR_THRESHOLD ? "#f44" : "#4f4";
      ctx.lineWidth = 2;
      for (const eye of [LEFT_EYE, RIGHT_EYE]) {
        ctx.beginPath();
        eye.forEach((i, k) => {
          const p = lm[i];
          const x = p.x * canvas.width;
          const y = p.y * canvas.height;
          if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
      }

      const now = performance.now();
      if (avg < EAR_THRESHOLD) {
        if (closedSince === null) closedSince = now;
        if (now - closedSince >= DROWSY_MS) alertDrowsy();
      } else {
        closedSince = null;
        clearAlert();
      }
    } else {
      statsEl.textContent = "EAR: no face";
      closedSince = null;
    }
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", init);
