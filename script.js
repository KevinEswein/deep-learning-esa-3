let model;
let wordIndex = {};
let indexWord = {};
let allWords = [];
let maxLength = 10;
let autoPredict = false;

async function loadData() {
    const response = await fetch('data.txt');
    const text = await response.text();
    allWords = text.split(/\s+/);
    createWordIndex(allWords);
}

function createWordIndex(words) {
    const uniqueWords = [...new Set(words)];
    uniqueWords.forEach((word, index) => {
        wordIndex[word] = index + 1;
        indexWord[index + 1] = word;
    });
}

function createModel() {
    model = tf.sequential();
    model.add(tf.layers.embedding({ inputDim: Object.keys(wordIndex).length + 1, outputDim: 100, inputLength: maxLength }));
    model.add(tf.layers.lstm({ units: 100, returnSequences: true, kernelInitializer: 'glorotUniform' }));
    model.add(tf.layers.lstm({ units: 100, kernelInitializer: 'glorotUniform' }));
    model.add(tf.layers.dense({ units: Object.keys(wordIndex).length + 1, activation: 'softmax' }));
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(0.01),
        metrics: ['accuracy']
    });
}

async function trainModel() {
    createModel();
    const sequences = [];
    const nextWords = [];
    for (let i = 0; i < allWords.length - maxLength; i++) {
        sequences.push(allWords.slice(i, i + maxLength));
        nextWords.push(allWords[i + maxLength]);
    }

    const xs = sequences.map(seq => seq.map(word => wordIndex[word] || 0));
    const ys = nextWords.map(word => wordIndex[word] || 0);

    const xsTensor = tf.tensor2d(xs);
    const ysTensor = tf.oneHot(tf.tensor1d(ys, 'int32'), Object.keys(wordIndex).length + 1);

    await model.fit(xsTensor, ysTensor, {
        epochs: 30,
        batchSize: 32,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'acc'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });

    xsTensor.dispose();
    ysTensor.dispose();
    document.getElementById('predictButton').disabled = false;
    document.getElementById('continueButton').disabled = false;
    document.getElementById('autoButton').disabled = false;
    document.getElementById('stopButton').disabled = false;
}

function predictNextWord(inputText) {
    const inputWords = inputText.trim().split(/\s+/);
    const inputSeq = inputWords.slice(-maxLength).map(word => wordIndex[word] || 0);
    const paddedSeq = Array(maxLength - inputSeq.length).fill(0).concat(inputSeq);
    const inputTensor = tf.tensor2d([paddedSeq]);

    const prediction = model.predict(inputTensor);
    const predictedIndex = prediction.argMax(-1).dataSync()[0];
    const predictedWord = indexWord[predictedIndex];

    return predictedWord;
}

document.getElementById('trainModelButton').addEventListener('click', async () => {
    await loadData();
    await trainModel();
});

document.getElementById('predictButton').addEventListener('click', () => {
    const inputText = document.getElementById('inputText').value;
    const predictedWord = predictNextWord(inputText);
    displayPredictions([predictedWord]);
});

document.getElementById('continueButton').addEventListener('click', () => {
    const inputText = document.getElementById('inputText').value;
    const predictedWord = predictNextWord(inputText);
    displayPredictions([predictedWord]);
    document.getElementById('inputText').value = inputText + ' ' + predictedWord;
});

document.getElementById('autoButton').addEventListener('click', () => {
    autoPredict = true;
    autoPredictWords();
});

document.getElementById('stopButton').addEventListener('click', () => {
    autoPredict = false;
});

document.getElementById('resetButton').addEventListener('click', () => {
    document.getElementById('inputText').value = '';
    document.getElementById('predictions').innerHTML = '';
});

function displayPredictions(predictions) {
    const predictionsDiv = document.getElementById('predictions');
    predictionsDiv.innerHTML = '';
    predictions.forEach(prediction => {
        const button = document.createElement('button');
        button.textContent = prediction;
        button.onclick = () => {
            document.getElementById('inputText').value += ' ' + prediction;
        };
        predictionsDiv.appendChild(button);
    });
}

async function autoPredictWords() {
    while (autoPredict) {
        const inputText = document.getElementById('inputText').value;
        const predictedWord = predictNextWord(inputText);
        document.getElementById('inputText').value += ' ' + predictedWord;
        displayPredictions([predictedWord]);
        await new Promise(resolve => setTimeout(resolve, 500)); // Warte 500ms zwischen den Vorhersagen
    }
}
