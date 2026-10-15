// MOJA Jersey Hijab — browser try-on (test build).
// Same hijab asset and shade list as the Snap Camera Kit version; face tracking here
// is MediaPipe Face Landmarker, rendering is three.js. Runs entirely on the device.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const WASM_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const params = new URLSearchParams(location.search);
const DEBUG_POSE = params.get("debug") === "pose";

// Hijab asset is in metres around the head centre; MediaPipe's face frame is in cm
// with its origin a few cm behind the face surface. BASE maps one onto the other.
const BASE = { y: 1.6, z: -3.4, s: 95 };
const DRAPE = { yawFollow: 0.3, tiltFollow: 0.45, gravityKeep: 0.55, bodyCatchUp: 1.2, pivotRatio: 0.55, smoothing: 16 };

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("gl");
const statusEl = $("status");

let shades = [];
let current = null;
let landmarker = null;
let renderer, scene, camera, hemi, sun;
let hijabRoot, fitGroup, hijabMat, headBone, chestBone, restOffset, restRelRot;
let lastSeen = 0;
let fit = loadFit();

// ---------------------------------------------------------------- UI helpers
function show(id, on) { $(id).hidden = !on; }
function message(title, body, action) {
  $("msg-title").textContent = title;
  $("msg-body").textContent = body;
  const b = $("msg-action");
  b.hidden = !action;
  if (action) { b.textContent = action.label; b.onclick = action.run; }
  show("intro", false); show("controls", false); show("message", true);
}
function setStatus(text) { statusEl.textContent = text || ""; statusEl.hidden = !text; }

function loadFit() {
  try { return Object.assign({ y: 0, z: 0, s: 1 }, JSON.parse(localStorage.getItem("moja-fit") || "{}")); }
  catch { return { y: 0, z: 0, s: 1 }; }
}
function saveFit() { try { localStorage.setItem("moja-fit", JSON.stringify(fit)); } catch {} }
function applyFit() {
  if (!fitGroup) return;
  fitGroup.position.set(0, BASE.y + fit.y, BASE.z + fit.z);
  fitGroup.scale.setScalar(BASE.s * fit.s);
}

function renderShades() {
  const wrap = $("shades");
  wrap.innerHTML = "";
  for (const s of shades) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "swatch"; b.dataset.id = s.id;
    b.setAttribute("role", "radio"); b.setAttribute("aria-label", s.name);
    b.style.setProperty("--c", s.hex);
    b.innerHTML = "<i></i>";
    b.onclick = () => selectShade(s);
    wrap.appendChild(b);
  }
}
function selectShade(s) {
  current = s;
  if (hijabMat) {
    hijabMat.color.set(s.hex);
    hijabMat.sheenColor.set(s.hex).lerp(new THREE.Color(1, 1, 1), 0.35);
  }
  $("shade-label").textContent = s.name;
  $("shop").href = s.url;
  $("shop").textContent = `Shop ${s.name}`;
  document.querySelectorAll(".swatch").forEach((b) => {
    const on = b.dataset.id === s.id;
    b.setAttribute("aria-checked", String(on));
    if (on) b.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
  });
}

// ---------------------------------------------------------------- 3D
function initThree() {
  renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, premultipliedAlpha: true });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.setPixelRatio(1);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(63, 9 / 16, 1, 10000); // MediaPipe's default face-geometry camera
  hemi = new THREE.HemisphereLight(0xfff6ec, 0x5a5048, 0.55);
  sun = new THREE.DirectionalLight(0xffffff, 0.75);
  sun.position.set(-30, 45, 60);
  scene.add(hemi, sun);
}

async function loadHijab() {
  const gltf = await new GLTFLoader().loadAsync("./moja_jersey_hijab.glb");
  const asset = gltf.scene;
  let mesh = null;
  asset.traverse((o) => { if (o.isSkinnedMesh || o.isMesh) mesh = mesh || o; });
  if (!mesh) throw new Error("hijab mesh missing");
  mesh.frustumCulled = false;
  const src = mesh.material;
  hijabMat = new THREE.MeshPhysicalMaterial({
    map: src.map || null,
    normalMap: src.normalMap || null,
    roughness: 0.86, metalness: 0,
    sheen: 0.55, sheenRoughness: 0.55,
    side: THREE.DoubleSide,
  });
  if (src.normalMap) hijabMat.normalScale.copy(src.normalScale).multiplyScalar(0.8);
  mesh.material = hijabMat;

  // Invisible head that writes depth only: hides the inside back of the hijab
  // when you look through the face opening.
  const occluder = new THREE.Mesh(
    new THREE.SphereGeometry(1, 32, 24),
    new THREE.MeshBasicMaterial({ colorWrite: false }),
  );
  occluder.scale.set(0.066, 0.095, 0.085);
  occluder.position.set(0, -0.012, 0.0);
  occluder.renderOrder = -1;

  fitGroup = new THREE.Group();
  fitGroup.add(asset, occluder);
  hijabRoot = new THREE.Group();
  hijabRoot.matrixAutoUpdate = false;
  hijabRoot.add(fitGroup);
  hijabRoot.visible = false;
  scene.add(hijabRoot);
  applyFit();

  headBone = asset.getObjectByName("HeadBone");
  chestBone = asset.getObjectByName("ChestBone");
  if (headBone && chestBone) {
    asset.updateMatrixWorld(true);
    const rel = headBone.matrixWorld.clone().invert().multiply(chestBone.matrixWorld);
    restOffset = new THREE.Vector3(); restRelRot = new THREE.Quaternion();
    rel.decompose(restOffset, restRelRot, new THREE.Vector3());
  }
}

// ---------------------------------------------------------------- pose
const faceM = new THREE.Matrix4();
const sm = { p: null, q: new THREE.Quaternion(), s: new THREE.Vector3() };
const tmp = { p: new THREE.Vector3(), q: new THREE.Quaternion(), s: new THREE.Vector3() };
const fitM = new THREE.Matrix4();

function setFacePose(matrixData, dt) {
  faceM.fromArray(matrixData);
  faceM.decompose(tmp.p, tmp.q, tmp.s);
  if (!sm.p) { sm.p = tmp.p.clone(); sm.q.copy(tmp.q); sm.s.copy(tmp.s); }
  // speed-adaptive smoothing: steady when still, low lag when moving
  const speed = tmp.p.distanceTo(sm.p) / Math.max(dt, 1e-3);
  const a = Math.min(1, 0.35 + speed * 0.004);
  sm.p.lerp(tmp.p, a); sm.q.slerp(tmp.q, a); sm.s.lerp(tmp.s, a);
  hijabRoot.matrix.compose(sm.p, sm.q, sm.s);
  hijabRoot.matrixWorldNeedsUpdate = true;
}

// Chest drape: same logic as lens-studio/Scripts/HijabDrapeRig.js
let torsoBase = null, drapeP = null, drapeQ = null;
const I = new THREE.Quaternion();
const v = { hp: new THREE.Vector3(), hs: new THREE.Vector3(), hq: new THREE.Quaternion() };
function updateDrape(dt) {
  if (!headBone || !chestBone) return;
  scene.updateMatrixWorld(true);
  headBone.matrixWorld.decompose(v.hp, v.hq, v.hs);
  if (!torsoBase) torsoBase = v.hq.clone();
  torsoBase.slerp(v.hq, 1 - Math.exp(-dt * DRAPE.bodyCatchUp));

  const rel = torsoBase.clone().invert().multiply(v.hq);
  const twist = new THREE.Quaternion(0, rel.y, 0, rel.w).normalize();
  const swing = rel.clone().multiply(twist.clone().invert());
  const partial = I.clone().slerp(swing, DRAPE.tiltFollow).multiply(I.clone().slerp(twist, DRAPE.yawFollow));
  let torso = torsoBase.clone().multiply(partial);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(torso);
  const g = new THREE.Quaternion().setFromUnitVectors(up, new THREE.Vector3(0, 1, 0));
  torso = I.clone().slerp(g, DRAPE.gravityKeep).multiply(torso);

  const off = restOffset.clone().multiply(v.hs);
  const pivotLocal = off.clone().multiplyScalar(DRAPE.pivotRatio);
  const pivotW = v.hp.clone().add(pivotLocal.clone().applyQuaternion(v.hq));
  const targetP = pivotW.add(off.clone().sub(pivotLocal).applyQuaternion(torso));
  const targetQ = torso.clone().multiply(restRelRot);

  const a = 1 - Math.exp(-dt * DRAPE.smoothing);
  if (!drapeP) { drapeP = targetP.clone(); drapeQ = targetQ.clone(); }
  drapeP.lerp(targetP, a); drapeQ.slerp(targetQ, a);

  const world = new THREE.Matrix4().compose(drapeP, drapeQ, v.hs);
  const local = chestBone.parent.matrixWorld.clone().invert().multiply(world);
  local.decompose(chestBone.position, chestBone.quaternion, chestBone.scale);
}
function resetDrape() { torsoBase = null; drapeP = null; drapeQ = null; sm.p = null; }

// ---------------------------------------------------------------- light from the room
const probe = document.createElement("canvas");
probe.width = probe.height = 24;
const probeCtx = probe.getContext("2d", { willReadFrequently: true });
let frameNo = 0, lightGain = 1;
function adaptLight() {
  if (frameNo++ % 20 || !video.videoWidth) return;
  probeCtx.drawImage(video, 0, 0, 24, 24);
  const d = probeCtx.getImageData(0, 0, 24, 24).data;
  let sum = 0;
  for (let i = 0; i < d.length; i += 4) sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
  const lum = sum / (d.length / 4) / 255;
  const target = Math.min(1.35, Math.max(0.55, 0.45 + lum * 1.3));
  lightGain += (target - lightGain) * 0.5;
  hemi.intensity = 0.55 * lightGain;
  sun.intensity = 0.75 * lightGain;
}

// ---------------------------------------------------------------- loop
function resize() {
  const w = video.videoWidth || 720, h = video.videoHeight || 1280;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

let lastVideoTime = -1, lastT = performance.now();
function frame() {
  const now = performance.now();
  const dt = Math.min((now - lastT) / 1000, 0.1);
  lastT = now;
  if (DEBUG_POSE) {
    const yaw = THREE.MathUtils.degToRad(Number(params.get("yaw") || 0) * Math.sin(now / 900));
    const m = new THREE.Matrix4().compose(new THREE.Vector3(0, -3, -48), new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, 0)), new THREE.Vector3(1, 1, 1));
    setFacePose(m.toArray(), dt);
    hijabRoot.visible = true;
  } else if (landmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const res = landmarker.detectForVideo(video, now);
    const mats = res.facialTransformationMatrixes;
    if (mats && mats.length) {
      if (now - lastSeen > 400) resetDrape();
      lastSeen = now;
      setFacePose(mats[0].data, dt);
      hijabRoot.visible = true;
      setStatus("");
    } else if (now - lastSeen > 250) {
      hijabRoot.visible = false;
      setStatus("Look at the camera");
    }
  }
  if (hijabRoot.visible) updateDrape(dt);
  adaptLight();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- startup
async function createLandmarker() {
  const files = await FilesetResolver.forVisionTasks(WASM_PATH);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", numFaces: 1,
    outputFacialTransformationMatrixes: true, outputFaceBlendshapes: false,
    minFaceDetectionConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await FaceLandmarker.createFromOptions(files, opts("GPU")); }
  catch (e) { console.warn("GPU delegate failed, using CPU", e); return FaceLandmarker.createFromOptions(files, opts("CPU")); }
}

async function start() {
  const btn = $("start");
  btn.disabled = true;
  btn.textContent = "Opening camera…";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
    });
  } catch (e) {
    const n = e && e.name;
    if (n === "NotAllowedError" || n === "SecurityError")
      return message("Camera permission needed", "Allow camera access for this site in your browser settings, then reload.", { label: "Reload", run: () => location.reload() });
    if (n === "NotFoundError" || n === "OverconstrainedError")
      return message("No front camera found", "Open this page on a phone or a computer with a webcam.");
    if (n === "NotReadableError")
      return message("Camera is busy", "Close other apps or tabs using the camera, then reload.", { label: "Reload", run: () => location.reload() });
    return message("Couldn't start the camera", String((e && e.message) || e));
  }
  video.srcObject = stream;
  await video.play();
  resize();
  video.addEventListener("resize", resize);
  show("intro", false);
  show("controls", true);
  $("fit-toggle").hidden = false;
  requestAnimationFrame(frame);

  if (!DEBUG_POSE) {
    setStatus("Loading face tracking…");
    try {
      landmarker = await createLandmarker();
      setStatus("Look at the camera");
    } catch (e) {
      console.error(e);
      message("Face tracking didn't load", "Check your connection and reload the page.", { label: "Reload", run: () => location.reload() });
    }
  }
}

function environmentProblem() {
  if (/Instagram|FBAN|FBAV|Snapchat|TikTok|musical_ly|Line\//i.test(navigator.userAgent))
    return ["Open in your browser", "This app's built-in browser can't use the camera. Tap ··· and choose “Open in Safari” or “Open in Chrome”."];
  if (!window.isSecureContext) return ["Secure connection needed", "The try-on only works on an https:// page."];
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
    return ["Camera not available", "Your browser doesn't allow camera access. Please update it or try Safari or Chrome."];
  if (!document.createElement("canvas").getContext("webgl2"))
    return ["Browser not supported", "Virtual try-on needs a newer browser (Safari 16+ or Chrome 95+)."];
  return null;
}

function wireFit() {
  const panel = $("fit"), toggle = $("fit-toggle");
  const inputs = { y: $("fit-y"), z: $("fit-z"), s: $("fit-s") };
  const sync = () => { for (const k in inputs) inputs[k].value = fit[k]; };
  sync();
  for (const k in inputs) inputs[k].addEventListener("input", () => { fit[k] = Number(inputs[k].value); applyFit(); saveFit(); });
  toggle.onclick = () => { panel.hidden = !panel.hidden; toggle.setAttribute("aria-expanded", String(!panel.hidden)); };
  $("fit-close").onclick = () => { panel.hidden = true; toggle.setAttribute("aria-expanded", "false"); };
  $("fit-reset").onclick = () => { fit = { y: 0, z: 0, s: 1 }; sync(); applyFit(); saveFit(); };
}

async function boot() {
  const problem = environmentProblem();
  if (problem) return message(problem[0], problem[1]);
  initThree();
  wireFit();
  $("start").onclick = start;
  try {
    const data = await (await fetch("./shades.json")).json();
    shades = data.shades;
    renderShades();
    await loadHijab();
    const want = params.get("shade") || data.defaultShade;
    selectShade(shades.find((s) => s.id === want) || shades[0]);
  } catch (e) {
    console.error(e);
    message("Try-on unavailable", "The hijab model didn't load. Check your connection and reload.", { label: "Reload", run: () => location.reload() });
  }
}

boot();
