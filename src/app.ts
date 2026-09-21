import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";
import "./style.css";

type Point = Readonly<{ x: number; y: number; z?: number }>;

const video = document.querySelector<HTMLVideoElement>("#video");
const warpCanvas = document.querySelector<HTMLCanvasElement>("#warp-canvas");
if (video === null || warpCanvas === null) {
  throw new Error("Pose stage elements are missing");
}

type TouchState = {
  targetX: number;
  targetY: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  active: boolean;
  strength: number;
};

type WarpGridField = {
  readonly cols: number;
  readonly rows: number;
  readonly offsetX: Float32Array;
  readonly offsetY: Float32Array;
  readonly velocityX: Float32Array;
  readonly velocityY: Float32Array;
  readonly pixels: Uint8Array;
};

type WebGLWarp = (tips: readonly Point[], timestamp: number) => void;

const fingertipIndices = [4, 8, 12, 16, 20] as const;
const previousTips: Point[] = [];
const warpStrengths = new Float32Array(10);
const screenTouch: TouchState = {
  targetX: -1e5,
  targetY: -1e5,
  x: -1e5,
  y: -1e5,
  vx: 0,
  vy: 0,
  active: false,
  strength: 0,
};
let previousFrameTime = 0;

const collectFingertips = (hands: readonly (readonly Point[])[]): readonly Point[] =>
  hands.flatMap((hand) => fingertipIndices.flatMap((index) => {
    const tip = hand[index];
    return tip === undefined ? [] : [tip];
  }));

const createWarpGridField = (cols: number, rows: number): WarpGridField => ({
  cols,
  rows,
  offsetX: new Float32Array(cols * rows),
  offsetY: new Float32Array(cols * rows),
  velocityX: new Float32Array(cols * rows),
  velocityY: new Float32Array(cols * rows),
  pixels: new Uint8Array(cols * rows * 4),
});

const applyWarpGridImpulse = (
  field: WarpGridField,
  x: number,
  y: number,
  velocityX: number,
  velocityY: number,
  width: number,
  height: number,
): void => {
  const radius = 150;
  const cellWidth = width / field.cols;
  const cellHeight = height / field.rows;
  for (let row = 0; row < field.rows; row += 1) {
    const cellY = (row + 0.5) * cellHeight;
    for (let col = 0; col < field.cols; col += 1) {
      const cellX = (col + 0.5) * cellWidth;
      const distance = Math.hypot(cellX - x, cellY - y);
      if (distance >= radius) continue;
      const falloff = 1 - distance / radius;
      const index = row * field.cols + col;
      field.velocityX[index] += velocityX * 0.08 * falloff;
      field.velocityY[index] += velocityY * 0.08 * falloff;
    }
  }
};

const integrateWarpGrid = (field: WarpGridField, maxShift: number): void => {
  for (let index = 0; index < field.offsetX.length; index += 1) {
    const velocityX = (field.velocityX[index] - field.offsetX[index] * 0.12) * 0.86;
    const velocityY = (field.velocityY[index] - field.offsetY[index] * 0.12) * 0.86;
    field.velocityX[index] = velocityX;
    field.velocityY[index] = velocityY;
    field.offsetX[index] = Math.max(-maxShift, Math.min(maxShift, field.offsetX[index] + velocityX));
    field.offsetY[index] = Math.max(-maxShift, Math.min(maxShift, field.offsetY[index] + velocityY));
    const pixelIndex = index * 4;
    field.pixels[pixelIndex] = Math.round((field.offsetX[index] / maxShift * 0.5 + 0.5) * 255);
    field.pixels[pixelIndex + 1] = Math.round((field.offsetY[index] / maxShift * 0.5 + 0.5) * 255);
    field.pixels[pixelIndex + 2] = 0;
    field.pixels[pixelIndex + 3] = 255;
  }
};

const isTouchingScreen = (tip: Point): boolean => (tip.z ?? 0) < -0.08;

const createShader = (gl: WebGLRenderingContext, type: number, source: string): WebGLShader => {
  const shader = gl.createShader(type);
  if (shader === null) throw new Error("Unable to create warp shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) ?? "Warp shader compilation failed");
  }
  return shader;
};

const createWebGLWarp = (): WebGLWarp => {
  const gl = warpCanvas.getContext("webgl", { alpha: false, antialias: false });
  if (gl === null) throw new Error("WebGL is unavailable");
  const grid = createWarpGridField(28, 18);
  const gridMaxShift = 60;
  const gridTexture = gl.createTexture();
  if (gridTexture === null) throw new Error("Unable to create warp grid texture");
  gl.bindTexture(gl.TEXTURE_2D, gridTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const vertex = createShader(gl, gl.VERTEX_SHADER, `
    attribute vec2 position;
    varying vec2 uv;
    void main() {
      uv = position * 0.5 + 0.5;
      gl_Position = vec4(position, 0.0, 1.0);
    }
  `);
  const fragment = createShader(gl, gl.FRAGMENT_SHADER, `
    precision highp float;
    uniform sampler2D camera;
    uniform sampler2D warpField;
    uniform vec2 resolution;
    uniform vec2 gridSize;
    uniform float gridMaxShift;
    uniform float gridLines;
    uniform vec3 gridColor;
    uniform vec2 tips[10];
    uniform float strengths[10];
    uniform int tipCount;
    uniform vec2 clothTouch;
    uniform vec2 clothVelocity;
    uniform float clothStrength;
    varying vec2 uv;

    vec3 sampleCamera(vec2 point) {
      return texture2D(camera, vec2(point.x, 1.0 - point.y)).rgb;
    }

    void main() {
      vec2 warped = uv;
      vec2 gridOffset = (texture2D(warpField, uv).rg - 0.5) * 2.0 * gridMaxShift;
      float gridShear = length(gridOffset) / max(gridMaxShift, 0.001);
      warped -= gridOffset / resolution;
      float total = gridShear;
      for (int i = 0; i < 10; i += 1) {
        if (i >= tipCount) break;
        vec2 delta = warped - tips[i];
        float distanceFromTip = length(delta);
        float influence = exp(-distanceFromTip * distanceFromTip / 0.018) * strengths[i];
        vec2 radial = distanceFromTip > 0.001 ? normalize(delta) : vec2(0.0);
        vec2 swirl = vec2(-delta.y, delta.x);
        warped += (radial * 0.045 + swirl * 0.22) * influence;
        total += influence;
      }

      // The screen touch acts like a soft cloth brush: it follows the finger
      // with spring-like lag and leaves a wider, directional wake in the image.
      vec2 clothDelta = warped - clothTouch;
      float clothDistance = dot(clothDelta, clothDelta);
      float clothInfluence = exp(-clothDistance / 0.055) * clothStrength;
      vec2 clothRadial = clothDistance > 0.0001 ? normalize(clothDelta) : vec2(0.0);
      vec2 clothSwirl = vec2(-clothDelta.y, clothDelta.x);
      warped += (clothRadial * 0.075 + clothSwirl * 0.3 + clothVelocity * 0.018) * clothInfluence;
      total += clothInfluence;

      // Split the colour channels only where pixels are actually moving.
      vec2 aberration = vec2(0.006 * total, 0.0);
      float red = sampleCamera(warped + aberration).r;
      float green = sampleCamera(warped).g;
      float blue = sampleCamera(warped - aberration).b;
      vec2 cellPosition = fract(warped * gridSize);
      vec2 cellSize = 1.0 / gridSize;
      vec2 lineDistance = min(cellPosition, 1.0 - cellPosition) * cellSize;
      float lineMask = 1.0 - smoothstep(0.0, 0.002, min(lineDistance.x, lineDistance.y));
      vec3 color = vec3(red, green, blue);
      color = mix(color, gridColor, lineMask * gridLines);
      gl_FragColor = vec4(color, 1.0);
    }
  `);
  const program = gl.createProgram();
  if (program === null) throw new Error("Unable to create warp program");
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) ?? "Warp program linking failed");
  }

  const buffer = gl.createBuffer();
  const texture = gl.createTexture();
  if (buffer === null || texture === null) throw new Error("Unable to create warp buffers");
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.useProgram(program);
  const position = gl.getAttribLocation(program, "position");
  const camera = gl.getUniformLocation(program, "camera");
  const warpField = gl.getUniformLocation(program, "warpField");
  const resolution = gl.getUniformLocation(program, "resolution");
  const gridSize = gl.getUniformLocation(program, "gridSize");
  const gridMaxShiftUniform = gl.getUniformLocation(program, "gridMaxShift");
  const gridLines = gl.getUniformLocation(program, "gridLines");
  const gridColor = gl.getUniformLocation(program, "gridColor");
  const tips = gl.getUniformLocation(program, "tips");
  const strengths = gl.getUniformLocation(program, "strengths");
  const tipCount = gl.getUniformLocation(program, "tipCount");
  const clothTouch = gl.getUniformLocation(program, "clothTouch");
  const clothVelocity = gl.getUniformLocation(program, "clothVelocity");
  const clothStrength = gl.getUniformLocation(program, "clothStrength");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.uniform1i(camera, 0);

  return (currentTips, timestamp) => {
    const frameDelta = previousFrameTime === 0 ? 1 / 60 : Math.min((timestamp - previousFrameTime) / 1000, 1 / 20);
    previousFrameTime = timestamp;
    const spring = 14;
    const targetStrength = screenTouch.active ? 1 : 0;
    const strengthRate = screenTouch.active ? 8 : 2.5;
    screenTouch.strength += (targetStrength - screenTouch.strength) * Math.min(frameDelta * strengthRate, 1);
    screenTouch.vx += ((screenTouch.targetX - screenTouch.x) * spring * spring - 2 * spring * screenTouch.vx) * frameDelta;
    screenTouch.vy += ((screenTouch.targetY - screenTouch.y) * spring * spring - 2 * spring * screenTouch.vy) * frameDelta;
    screenTouch.x += screenTouch.vx * frameDelta;
    screenTouch.y += screenTouch.vy * frameDelta;

    const tipData = new Float32Array(20);
    const strengthData = new Float32Array(10);
    let visibleCount = 0;
    currentTips.forEach((tip, index) => {
      const previous = previousTips[index];
      const speed = previous === undefined ? 0 : Math.hypot(tip.x - previous.x, tip.y - previous.y);
      const active = isTouchingScreen(tip);
      const target = active ? Math.min(1, speed * 55) : 0;
      warpStrengths[index] = warpStrengths[index] * 0.86 + target * 0.14;
      tipData[index * 2] = tip.x;
      tipData[index * 2 + 1] = 1 - tip.y;
      strengthData[index] = warpStrengths[index];
      if (active) visibleCount += 1;
    });
    previousTips.splice(0, previousTips.length, ...currentTips);
    gl.canvas.width = warpCanvas.clientWidth * window.devicePixelRatio;
    gl.canvas.height = warpCanvas.clientHeight * window.devicePixelRatio;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    const canvasWidth = Math.max(warpCanvas.clientWidth, 1);
    const canvasHeight = Math.max(warpCanvas.clientHeight, 1);
    if (screenTouch.active) {
      applyWarpGridImpulse(
        grid,
        screenTouch.x * canvasWidth,
        screenTouch.y * canvasHeight,
        screenTouch.vx * canvasWidth,
        screenTouch.vy * canvasHeight,
        canvasWidth,
        canvasHeight,
      );
    }
    integrateWarpGrid(grid, gridMaxShift);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, gridTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, grid.cols, grid.rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, grid.pixels);
    gl.useProgram(program);
    gl.uniform2fv(tips, tipData);
    gl.uniform1fv(strengths, strengthData);
    gl.uniform1i(tipCount, Math.min(currentTips.length, 10));
    gl.uniform1i(camera, 0);
    gl.uniform1i(warpField, 1);
    gl.uniform2f(resolution, canvasWidth, canvasHeight);
    gl.uniform2f(gridSize, grid.cols, grid.rows);
    gl.uniform1f(gridMaxShiftUniform, gridMaxShift);
    gl.uniform1f(gridLines, 0.0);
    gl.uniform3f(gridColor, 1, 1, 1);
    gl.uniform2f(clothTouch, screenTouch.x, screenTouch.y);
    gl.uniform2f(clothVelocity, screenTouch.vx, screenTouch.vy);
    gl.uniform1f(clothStrength, screenTouch.strength);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    return visibleCount;
  };
};

const resizeCanvas = (): void => {
  const bounds = warpCanvas.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  warpCanvas.width = Math.round(bounds.width * pixelRatio);
  warpCanvas.height = Math.round(bounds.height * pixelRatio);
};

const updateScreenTouch = (event: PointerEvent): void => {
  const bounds = warpCanvas.getBoundingClientRect();
  const x = 1 - (event.clientX - bounds.left) / Math.max(bounds.width, 1);
  const y = 1 - (event.clientY - bounds.top) / Math.max(bounds.height, 1);
  screenTouch.targetX = Math.min(1, Math.max(0, x));
  screenTouch.targetY = Math.min(1, Math.max(0, y));
  screenTouch.active = true;
};

const releaseScreenTouch = (): void => {
  screenTouch.active = false;
  screenTouch.targetX = -1e5;
  screenTouch.targetY = -1e5;
};

const run = async (): Promise<void> => {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  warpCanvas.addEventListener("pointerdown", updateScreenTouch);
  warpCanvas.addEventListener("pointermove", updateScreenTouch);
  warpCanvas.addEventListener("pointerup", releaseScreenTouch);
  warpCanvas.addEventListener("pointercancel", releaseScreenTouch);
  warpCanvas.addEventListener("pointerleave", releaseScreenTouch);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 1920 },
      height: { ideal: 1080 },
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  const renderWarp = createWebGLWarp();
  const resolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm",
  );
  const [poseLandmarker, handLandmarker, faceLandmarker] = await Promise.all([
    PoseLandmarker.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", delegate: "GPU" },
      runningMode: "VIDEO", numPoses: 1,
    }),
    HandLandmarker.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO", numHands: 2,
    }),
    FaceLandmarker.createFromOptions(resolver, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO", numFaces: 1,
    }),
  ]);

  const render = (timestamp: number): void => {
    poseLandmarker.detectForVideo(video, timestamp);
    const handResult = handLandmarker.detectForVideo(video, timestamp);
    faceLandmarker.detectForVideo(video, timestamp);
    const fingertips = collectFingertips(handResult.landmarks);
    renderWarp(fingertips, timestamp);
    requestAnimationFrame(render);
  };

  requestAnimationFrame(render);
};

run().catch((error: unknown) => {
  console.error(error);
});
