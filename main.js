import { AutoModel, AutoProcessor, RawImage } from "@huggingface/transformers";

const status = document.getElementById("status");
const container = document.getElementById("container");
const overlay = document.getElementById("overlay");
const canvas = document.getElementById("canvas");
const video = document.getElementById("video");

const thresholdSlider = document.getElementById("threshold");
const thresholdLabel = document.getElementById("threshold-value");
const sizeSlider = document.getElementById("size");
const sizeLabel = document.getElementById("size-value");
const scaleSlider = document.getElementById("scale");
const scaleLabel = document.getElementById("scale-value");
const flipCheckbox = document.getElementById("flip-horizontal");

// novos controles
const brightnessSlider = document.getElementById("brightness");
const brightnessLabel = document.getElementById("brightness-value");
const contrastSlider = document.getElementById("contrast");
const contrastLabel = document.getElementById("contrast-value");
const saturationSlider = document.getElementById("saturation");
const saturationLabel = document.getElementById("saturation-value");

function setStreamSize(width, height) {
  video.width = canvas.width = Math.round(width);
  video.height = canvas.height = Math.round(height);
}

status.textContent = "Loading model...";

const model_id = "Xenova/gelan-c_all";
const model = await AutoModel.from_pretrained(model_id);
const processor = await AutoProcessor.from_pretrained(model_id);

let scale = 0.5;
scaleSlider.addEventListener("input", () => {
  scale = Number(scaleSlider.value);
  setStreamSize(video.videoWidth * scale, video.videoHeight * scale);
  scaleLabel.textContent = scale;
});
scaleSlider.disabled = false;

let threshold = 0.25;
thresholdSlider.addEventListener("input", () => {
  threshold = Number(thresholdSlider.value);
  thresholdLabel.textContent = threshold.toFixed(2);
});
thresholdSlider.disabled = false;

let size = 128;
processor.feature_extractor.size = { shortest_edge: size };
sizeSlider.addEventListener("input", () => {
  size = Number(sizeSlider.value);
  processor.feature_extractor.size = { shortest_edge: size };
  sizeLabel.textContent = size;
});
sizeSlider.disabled = false;

let isFlipped = false;
flipCheckbox.addEventListener("change", () => {
  isFlipped = flipCheckbox.checked;
});

// vars brilho/contraste/saturação
let brightness = 1, contrast = 1, saturation = 1;

brightnessSlider.addEventListener("input", e => {
  brightness = e.target.value;
  brightnessLabel.textContent = brightness;
});

contrastSlider.addEventListener("input", e => {
  contrast = e.target.value;
  contrastLabel.textContent = contrast;
});

saturationSlider.addEventListener("input", e => {
  saturation = e.target.value;
  saturationLabel.textContent = saturation;
});

status.textContent = "Ready";

const COLOURS = [
  "#EF4444", "#4299E1", "#059669", "#FBBF24", "#4B52B1",
  "#7B3AC2", "#ED507A", "#1DD1A1", "#F3873A", "#4B5563",
  "#DC2626", "#1852B4", "#18A35D", "#F59E0B", "#4059BE",
  "#6027A5", "#D63D60", "#00AC9B", "#E64A19", "#272A34",
];

function renderBox([xmin, ymin, xmax, ymax, score, id], [w, h]) {
  if (score < threshold) return;
  if (isFlipped) {
    const temp = xmin;
    xmin = w - xmax;
    xmax = w - temp;
  }

  const color = COLOURS[id % COLOURS.length];
  const boxElement = document.createElement("div");
  boxElement.className = "bounding-box";
  Object.assign(boxElement.style, {
    borderColor: color,
    left: (100 * xmin) / w + "%",
    top: (100 * ymin) / h + "%",
    width: (100 * (xmax - xmin)) / w + "%",
    height: (100 * (ymax - ymin)) / h + "%",
  });

  const labelElement = document.createElement("span");
  labelElement.textContent = `${model.config.id2label[id]} (${(100 * score).toFixed(2)}%)`;
  labelElement.className = "bounding-box-label";
  labelElement.style.backgroundColor = color;

  boxElement.appendChild(labelElement);
  overlay.appendChild(boxElement);
}

let isProcessing = false;
let previousTime;
const context = canvas.getContext("2d", { willReadFrequently: true });

function updateCanvas() {
  const { width, height } = canvas;

  context.save();
  if (isFlipped) {
    context.scale(-1, 1);
    context.translate(-width, 0);
  }

  // aplica filtros
  context.filter = `brightness(${brightness}) contrast(${contrast}) saturate(${saturation})`;
  context.drawImage(video, 0, 0, width, height);
  context.restore();

  if (!isProcessing) {
    isProcessing = true;
    (async function () {
      const pixelData = context.getImageData(0, 0, width, height).data;
      const image = new RawImage(pixelData, width, height, 4);

      const inputs = await processor(image);
      const { outputs } = await model(inputs);

      overlay.innerHTML = "";
      const sizes = inputs.reshaped_input_sizes[0].reverse();
      outputs.tolist().forEach((x) => renderBox(x, sizes));

      if (previousTime !== undefined) {
        const fps = 1000 / (performance.now() - previousTime);
        status.textContent = `FPS: ${fps.toFixed(2)}`;
      }
      previousTime = performance.now();
      isProcessing = false;
    })();
  }

  window.requestAnimationFrame(updateCanvas);
}

navigator.mediaDevices
  .getUserMedia({ video: true })
  .then((stream) => {
    video.srcObject = stream;
    video.play();
    video.style.display = "none";

    const videoTrack = stream.getVideoTracks()[0];
    const { width, height } = videoTrack.getSettings();
    setStreamSize(width * scale, height * scale);

    const ar = width / height;
    const [cw, ch] = ar > 720 / 405 ? [720, 720 / ar] : [405 * ar, 405];
    container.style.width = `${cw}px`;
    container.style.height = `${ch}px`;

    window.requestAnimationFrame(updateCanvas);
  })
  .catch((error) => {
    alert(error);
  });
