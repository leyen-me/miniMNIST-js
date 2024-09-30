const fs = require("fs").promises;
const path = require("path");

// Constants
const INPUT_SIZE = 784;
const HIDDEN_SIZE = 256;
const OUTPUT_SIZE = 10;
const LEARNING_RATE = 0.001;
const EPOCHS = 20;
const BATCH_SIZE = 64;
const IMAGE_SIZE = 28;
const TRAIN_SPLIT = 0.8;

// Layer class definition
class Layer {
  constructor(inputSize, outputSize) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weights = Array.from(
      { length: inputSize * outputSize },
      () => (Math.random() - 0.5) * 2 * Math.sqrt(2.0 / inputSize)
    );
    this.biases = new Array(outputSize).fill(0);
  }
}

// Network class definition
class Network {
  constructor() {
    this.hidden = new Layer(INPUT_SIZE, HIDDEN_SIZE);
    this.output = new Layer(HIDDEN_SIZE, OUTPUT_SIZE);
  }
}

// Softmax activation function
function softmax(input) {
  const max = Math.max(...input);
  const exp = input.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map((x) => x / sum);
}

// Forward propagation
function forward(layer, input) {
  return layer.biases.map((bias, i) =>
    input.reduce(
      (sum, x, j) => sum + x * layer.weights[j * layer.outputSize + i],
      bias
    )
  );
}

// Backward propagation
function backward(layer, input, outputGrad, inputGrad, lr) {
  for (let i = 0; i < layer.outputSize; i++) {
    for (let j = 0; j < layer.inputSize; j++) {
      const idx = j * layer.outputSize + i;
      const grad = outputGrad[i] * input[j];
      layer.weights[idx] -= lr * grad;
      if (inputGrad) {
        inputGrad[j] += outputGrad[i] * layer.weights[idx];
      }
    }
    layer.biases[i] -= lr * outputGrad[i];
  }
}

// Training function
function train(net, input, label, lr) {
  let hiddenOutput = forward(net.hidden, input).map((x) => Math.max(0, x)); // ReLU
  let finalOutput = softmax(forward(net.output, hiddenOutput));

  const outputGrad = finalOutput.map((x, i) => x - (i === label ? 1 : 0));
  const hiddenGrad = new Array(HIDDEN_SIZE).fill(0);

  backward(net.output, hiddenOutput, outputGrad, hiddenGrad, lr);
  hiddenGrad.forEach(
    (grad, i) => (hiddenGrad[i] *= hiddenOutput[i] > 0 ? 1 : 0)
  ); // ReLU derivative
  backward(net.hidden, input, hiddenGrad, null, lr);
}

// Prediction function
function predict(net, input) {
  let hiddenOutput = forward(net.hidden, input).map((x) => Math.max(0, x)); // ReLU
  let finalOutput = softmax(forward(net.output, hiddenOutput));
  return finalOutput.indexOf(Math.max(...finalOutput));
}

// Load MNIST data
async function loadMNISTData() {
  const imagesPath = path.join(__dirname, "data", "train-images.idx3-ubyte");
  const labelsPath = path.join(__dirname, "data", "train-labels.idx1-ubyte");

  const imagesBuffer = await fs.readFile(imagesPath);
  const labelsBuffer = await fs.readFile(labelsPath);

  const imagesMagic = imagesBuffer.readUInt32BE(0);
  const imagesCount = imagesBuffer.readUInt32BE(4);
  const imagesRows = imagesBuffer.readUInt32BE(8);
  const imagesCols = imagesBuffer.readUInt32BE(12);
  const imagesOffset = 16;

  if (
    imagesMagic !== 2051 ||
    imagesRows !== IMAGE_SIZE ||
    imagesCols !== IMAGE_SIZE
  ) {
    throw new Error("Invalid image file");
  }

  const labelsMagic = labelsBuffer.readUInt32BE(0);
  const labelsCount = labelsBuffer.readUInt32BE(4);
  const labelsOffset = 8;

  if (labelsMagic !== 2049 || imagesCount !== labelsCount) {
    throw new Error("Invalid label file");
  }

  const images = Array.from(
    { length: imagesCount },
    (_, i) =>
      new Uint8Array(
        imagesBuffer.slice(
          imagesOffset + i * INPUT_SIZE,
          imagesOffset + (i + 1) * INPUT_SIZE
        )
      )
  );

  const labels = Array.from(
    { length: labelsCount },
    (_, i) => labelsBuffer[labelsOffset + i]
  );

  return { images, labels };
}

// Shuffle data
function shuffleData(images, labels) {
  for (let i = images.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [images[i], images[j]] = [images[j], images[i]];
    [labels[i], labels[j]] = [labels[j], labels[i]];
  }
}

// Main function
async function main() {
  const net = new Network();
  const data = await loadMNISTData();
  const learningRate = LEARNING_RATE;

  shuffleData(data.images, data.labels);

  const trainSize = Math.floor(data.images.length * TRAIN_SPLIT);
  const testSize = data.images.length - trainSize;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    let totalLoss = 0;
    for (let i = 0; i < trainSize; i += BATCH_SIZE) {
      for (let j = 0; j < BATCH_SIZE && i + j < trainSize; j++) {
        const idx = i + j;
        const img = data.images[idx].map((x) => x / 255.0);
        train(net, img, data.labels[idx], learningRate);

        const hiddenOutput = forward(net.hidden, img).map((x) =>
          Math.max(0, x)
        );
        const finalOutput = softmax(forward(net.output, hiddenOutput));
        totalLoss += -Math.log(finalOutput[data.labels[idx]] + 1e-10);
      }
    }

    let correct = 0;
    for (let i = trainSize; i < data.images.length; i++) {
      const img = data.images[i].map((x) => x / 255.0);
      if (predict(net, img) === data.labels[i]) {
        correct++;
      }
    }

    console.log(
      `Epoch ${epoch + 1}, Accuracy: ${((correct / testSize) * 100).toFixed(
        2
      )}%, Avg Loss: ${(totalLoss / trainSize).toFixed(4)}`
    );
  }
}

// Run main function
main().catch(console.error);
