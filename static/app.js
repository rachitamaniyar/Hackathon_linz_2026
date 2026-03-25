import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("webcam");
const captureCanvas = document.getElementById("captureCanvas");
const captureContext = captureCanvas.getContext("2d");

const openCameraButton = document.getElementById("openCameraButton");
const uploadImageButton = document.getElementById("uploadImageButton");
const startTimerButton = document.getElementById("startTimerButton");
const resetButton = document.getElementById("resetButton");
const imageUploadInput = document.getElementById("imageUploadInput");

const timerDisplay = document.getElementById("timerDisplay");
const timerStatus = document.getElementById("timerStatus");
const cameraState = document.getElementById("cameraState");
const resultStatus = document.getElementById("resultStatus");
const predictedPose = document.getElementById("predictedPose");
const confidenceValue = document.getElementById("confidenceValue");
const poseScoreValue = document.getElementById("poseScoreValue");
const summaryText = document.getElementById("summaryText");
const suggestionsList = document.getElementById("suggestionsList");

const COUNTDOWN_SECONDS = 10;
const MODEL_ASSET_PATH =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

let poseLandmarker = null;
let mediaPipeReady = false;
let stream = null;
let countdownValue = COUNTDOWN_SECONDS;
let countdownInterval = null;
let currentMode = "idle";

function setTimerState(value, statusText) {
  timerDisplay.textContent = String(value);
  timerStatus.textContent = statusText;
}

function setSuggestions(items) {
  suggestionsList.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    suggestionsList.appendChild(li);
  });
}

function resetResults() {
  predictedPose.textContent = "--";
  confidenceValue.textContent = "--";
  poseScoreValue.textContent = "--";
  summaryText.textContent = "The app will analyze a single frame after the timer reaches zero.";
  resultStatus.textContent = "No capture yet";
  setSuggestions(["Open the webcam to begin."]);
}

async function setupMediaPipe() {
  if (mediaPipeReady) {
    return;
  }

  resultStatus.textContent = "Loading MediaPipe pose detector";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_ASSET_PATH,
    },
    runningMode: "IMAGE",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
  });

  mediaPipeReady = true;
  resultStatus.textContent = "MediaPipe ready";
}

async function openCamera() {
  await setupMediaPipe();

  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 960 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  video.srcObject = stream;
  await video.play();

  captureCanvas.hidden = true;
  video.hidden = false;
  currentMode = "webcam";
  cameraState.textContent = "Live webcam ready";
  startTimerButton.disabled = false;
  resetButton.disabled = false;
  openCameraButton.disabled = true;
  uploadImageButton.disabled = true;
  setTimerState(COUNTDOWN_SECONDS, "Press start when your pose is ready.");
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
}

function resizeCaptureCanvas() {
  captureCanvas.width = video.videoWidth || 960;
  captureCanvas.height = video.videoHeight || 720;
}

function serializeLandmarks(landmarks) {
  return landmarks.map((landmark, index) => ({
    index,
    x: Number(landmark.x.toFixed(6)),
    y: Number(landmark.y.toFixed(6)),
    z: Number(landmark.z.toFixed(6)),
    visibility: Number((landmark.visibility ?? 0).toFixed(6)),
  }));
}

function drawCapturedPose(landmarks) {
  const drawingUtils = new DrawingUtils(captureContext);
  const connections = PoseLandmarker.POSE_CONNECTIONS || [];

  drawingUtils.drawConnectors(landmarks, connections, {
    color: "#ffb08f",
    lineWidth: 4,
  });

  drawingUtils.drawLandmarks(landmarks, {
    color: "#ffffff",
    lineWidth: 2,
    radius: 4,
  });
}

async function loadImageElement(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Uploaded image could not be loaded."));
    image.src = dataUrl;
  });
}

async function analyzeStillImage(imageSource, imageBase64, sourceLabel) {
  const poseResult = poseLandmarker.detect(imageSource);
  const rawLandmarks = poseResult.landmarks?.[0] || [];
  const landmarks = rawLandmarks.length ? serializeLandmarks(rawLandmarks) : [];

  if (!landmarks.length) {
    predictedPose.textContent = "--";
    confidenceValue.textContent = "--";
    poseScoreValue.textContent = "--";
    resultStatus.textContent = "No body landmarks detected";
    summaryText.textContent = "Retake or upload an image with your full body visible in the frame.";
    setSuggestions([
      "Use an image where the full pose is visible from head to feet.",
      "Avoid cluttered backgrounds and make sure the pose is clearly visible.",
    ]);
    return;
  }

  drawCapturedPose(rawLandmarks);
  resultStatus.textContent = "Sending pose to backend";

  const response = await fetch("/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      source: sourceLabel,
      image_base64: imageBase64,
      landmarks,
    }),
  });

  if (!response.ok) {
    throw new Error(`Backend returned ${response.status}`);
  }

  const result = await response.json();
  renderResult(result);
}

async function captureAndAnalyze() {
  resizeCaptureCanvas();
  captureContext.save();
  captureContext.scale(-1, 1);
  captureContext.drawImage(video, -captureCanvas.width, 0, captureCanvas.width, captureCanvas.height);
  captureContext.restore();

  const capturedImage = captureCanvas.toDataURL("image/jpeg", 0.9);

  video.hidden = true;
  captureCanvas.hidden = false;
  resultStatus.textContent = "Detecting body landmarks";
  cameraState.textContent = "Captured frame frozen for analysis";
  await analyzeStillImage(captureCanvas, capturedImage, "webcam-timer-capture");
  stopCamera();
}

function renderResult(result) {
  resultStatus.textContent = result.status === "determined" ? "Analysis complete" : "Analysis complete with low confidence";
  predictedPose.textContent = result.label || "Pose cannot be determined";
  confidenceValue.textContent = `${((result.confidence || 0) * 100).toFixed(1)}%`;
  poseScoreValue.textContent = `${result.pose_score ?? 0}/100`;
  summaryText.textContent = result.summary || result.message || "Analysis completed.";
  setSuggestions(result.suggestions?.length ? result.suggestions : ["No correction suggestions returned."]);
}

function startCountdown() {
  if (!stream) {
    return;
  }

  startTimerButton.disabled = true;
  countdownValue = COUNTDOWN_SECONDS;
  setTimerState(countdownValue, "Hold the pose until the timer ends.");
  resultStatus.textContent = "Timer running";

  countdownInterval = window.setInterval(async () => {
    countdownValue -= 1;
    setTimerState(Math.max(0, countdownValue), "Hold still. Capture happens at zero.");

    if (countdownValue > 0) {
      return;
    }

    window.clearInterval(countdownInterval);
    countdownInterval = null;
    setTimerState(0, "Analyzing captured frame.");

    try {
      await captureAndAnalyze();
    } catch (error) {
      resultStatus.textContent = "Analysis failed";
      summaryText.textContent = String(error);
      setSuggestions(["Check that the backend is running and try again."]);
      startTimerButton.disabled = false;
    }
  }, 1000);
}

function resetApp() {
  if (countdownInterval) {
    window.clearInterval(countdownInterval);
    countdownInterval = null;
  }

  stopCamera();
  resetResults();
  setTimerState(COUNTDOWN_SECONDS, "Open the webcam, then start the timer.");
  captureContext.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
  captureCanvas.hidden = true;
  video.hidden = false;
  currentMode = "idle";
  openCameraButton.disabled = false;
  uploadImageButton.disabled = false;
  startTimerButton.disabled = true;
  resetButton.disabled = true;
  cameraState.textContent = "Camera idle";
  imageUploadInput.value = "";
}

openCameraButton.addEventListener("click", async () => {
  try {
    await openCamera();
  } catch (error) {
    cameraState.textContent = "Camera access failed";
    summaryText.textContent = String(error);
    setSuggestions(["Allow camera permission in the browser and retry."]);
  }
});

uploadImageButton.addEventListener("click", async () => {
  try {
    await setupMediaPipe();
    imageUploadInput.click();
  } catch (error) {
    resultStatus.textContent = "Upload unavailable";
    summaryText.textContent = String(error);
    setSuggestions(["Reload the page and try uploading the image again."]);
  }
});

imageUploadInput.addEventListener("change", async (event) => {
  const [file] = event.target.files || [];
  if (!file) {
    return;
  }

  try {
    if (countdownInterval) {
      window.clearInterval(countdownInterval);
      countdownInterval = null;
    }

    stopCamera();
    resetResults();
    currentMode = "upload";
    openCameraButton.disabled = true;
    uploadImageButton.disabled = true;
    startTimerButton.disabled = true;
    resetButton.disabled = false;
    setTimerState("-", "Timer skipped for uploaded image.");
    cameraState.textContent = `Uploaded image: ${file.name}`;
    resultStatus.textContent = "Preparing uploaded image";

    const imageBase64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("Image file could not be read."));
      reader.readAsDataURL(file);
    });

    const image = await loadImageElement(imageBase64);
    captureCanvas.width = image.naturalWidth || image.width;
    captureCanvas.height = image.naturalHeight || image.height;
    captureContext.clearRect(0, 0, captureCanvas.width, captureCanvas.height);
    captureContext.drawImage(image, 0, 0, captureCanvas.width, captureCanvas.height);

    video.hidden = true;
    captureCanvas.hidden = false;

    await analyzeStillImage(captureCanvas, imageBase64, "uploaded-image");
  } catch (error) {
    resultStatus.textContent = "Upload analysis failed";
    summaryText.textContent = String(error);
    setSuggestions(["Try another image with the full body visible."]);
    uploadImageButton.disabled = false;
  }
});

startTimerButton.addEventListener("click", startCountdown);
resetButton.addEventListener("click", resetApp);

resetResults();
