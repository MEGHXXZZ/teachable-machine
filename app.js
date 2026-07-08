const webcamElement = document.getElementById('webcam');
const labelElement = document.getElementById('label');
const confidenceElement = document.getElementById('confidence');
const classesContainer = document.getElementById('classes-container');
const addClassBtn = document.getElementById('add-class-btn');
const classNameInput = document.getElementById('class-name-input');

const classifier = knnClassifier.create();

let classNames = [];
let mobilenetModel;
let isPredicting = false;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const hands = new Hands({
  locateFile: (file) =>
    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

hands.onResults(async (results) => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {

    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
        color: '#c084fc',
        lineWidth: 2
      });
      drawLandmarks(ctx, landmarks, {
        color: '#f472b6',
        lineWidth: 1,
        radius: 4
      });
    }

    if (isPredicting && classifier.getNumClasses() > 0) {
      const img = tf.browser.fromPixels(webcamElement);
      const features = mobilenetModel.infer(img, true);
      const result = await classifier.predictClass(features);

      const numHands = results.multiHandLandmarks.length;
      if (numHands === 1) {
        labelElement.innerText = classNames[result.label];
        confidenceElement.innerText =
          `Confidence: ${(result.confidences[result.label] * 100).toFixed(1)}%`;
      } else {
        labelElement.innerText = `Both hands: ${classNames[result.label]}`;
        confidenceElement.innerText =
          `Confidence: ${(result.confidences[result.label] * 100).toFixed(1)}%`;
      }
      img.dispose();
    }

  } else {
    if (isPredicting) {
      labelElement.innerText = 'No hand detected';
      confidenceElement.innerText = '';
    }
  }
});

async function main() {
  console.log('Loading MobileNet...');
  mobilenetModel = await mobilenet.load();
  console.log('MobileNet loaded!');

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  webcamElement.srcObject = stream;

  await new Promise((resolve) => {
    webcamElement.onloadedmetadata = () => {
      webcamElement.play();
      resolve();
    };
  });

  canvas.width = webcamElement.videoWidth;
  canvas.height = webcamElement.videoHeight;
  canvas.style.width = '320px';
  canvas.style.height = '240px';
  webcamElement.style.width = '320px';
  webcamElement.style.height = '240px';

  console.log(`Video size: ${webcamElement.videoWidth}x${webcamElement.videoHeight}`);

  const camera = new Camera(webcamElement, {
    onFrame: async () => {
      await hands.send({ image: webcamElement });
    },
    width: webcamElement.videoWidth,
    height: webcamElement.videoHeight
  });

  camera.start();
  console.log('Camera and MediaPipe started!');

  setTimeout(() => {
  console.log('Testing hands.onResults — is it firing?');
  hands.send({ image: webcamElement });
}, 2000);

}

main();

addClassBtn.addEventListener('click', () => {
  const className = classNameInput.value.trim();
  if (!className) return;

  const classIndex = classNames.length;
  classNames.push(className);
  classNameInput.value = '';

  const card = document.createElement('div');
  card.className = 'class-card';
  card.innerHTML = `
    <h3>${className}</h3>
    <p id="count-${classIndex}">0 samples</p>
    <button id="record-${classIndex}">Record ${className}</button>
  `;
  classesContainer.appendChild(card);

  document.getElementById(`record-${classIndex}`)
    .addEventListener('click', async () => {
      let count = 0;
      for (let i = 0; i < 30; i++) {
        const img = tf.browser.fromPixels(webcamElement);
        const features = mobilenetModel.infer(img, true);
        classifier.addExample(features, classIndex);
        img.dispose();
        count++;
        document.getElementById(`count-${classIndex}`)
          .innerText = `${count} samples`;
        await tf.nextFrame();
      }
      console.log(`${className} recorded!`);
    });
});

document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  console.log('Predicting...');
});