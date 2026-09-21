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

type WebGLWarp = (tips: readonly Point[]) => void;

const fingertipIndices = [4, 8, 12, 16, 20] as const;
const previousTips: Point[] = [];
const warpStrengths = new Float32Array(10);

const collectFingertips = (hands: readonly (readonly Point[])[]): readonly Point[] =>
  hands.flatMap((hand) => fingertipIndices.flatMap((index) => {
    const tip = hand[index];
    return tip === undefined ? [] : [tip];
  }));

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
    uniform vec2 tips[10];
    uniform float strengths[10];
    uniform int tipCount;
    varying vec2 uv;

    vec3 sampleCamera(vec2 point) {
      return texture2D(camera, vec2(point.x, 1.0 - point.y)).rgb;
    }

    void main() {
      vec2 warped = uv;
      float total = 0.0;
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

      // Split the colour channels only where pixels are actually moving.
      vec2 aberration = vec2(0.006 * total, 0.0);
      float red = sampleCamera(warped + aberration).r;
      float green = sampleCamera(warped).g;
      float blue = sampleCamera(warped - aberration).b;
      gl_FragColor = vec4(red, green, blue, 1.0);
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
  const tips = gl.getUniformLocation(program, "tips");
  const strengths = gl.getUniformLocation(program, "strengths");
  const tipCount = gl.getUniformLocation(program, "tipCount");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.uniform1i(camera, 0);

  return (currentTips) => {
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
      tipData[index * 2 + 1] = tip.y;
      strengthData[index] = warpStrengths[index];
      if (active) visibleCount += 1;
    });
    previousTips.splice(0, previousTips.length, ...currentTips);
    gl.canvas.width = warpCanvas.clientWidth * window.devicePixelRatio;
    gl.canvas.height = warpCanvas.clientHeight * window.devicePixelRatio;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.useProgram(program);
    gl.uniform2fv(tips, tipData);
    gl.uniform1fv(strengths, strengthData);
    gl.uniform1i(tipCount, Math.min(currentTips.length, 10));
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

const run = async (): Promise<void> => {
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
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
    const touchingTips = renderWarp(fingertips);
    requestAnimationFrame(render);
  };

  requestAnimationFrame(render);
};

run().catch((error: unknown) => {
  console.error(error);
});
