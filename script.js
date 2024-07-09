let model;
let tokenIndex;
let seqLength = 5;

async function loadData() {
    const response = await fetch('data.txt');
    const data = await response.text();
    return data.split(' ');
}

function createSequences(data, seqLength) {
    const sequences = [];
    for (let i = 0; i < data.length - seqLength; i++) {
        sequences.push(data.slice(i, i + seqLength + 1));
    }
    return sequences;
}

function createTokenIndex(data) {
    const uniqueTokens = Array.from(new Set(data));
    const tokenIndex = {};
    uniqueTokens.forEach((token, index) => {
        tokenIndex[token] = index;
    });
    return tokenIndex;
}

function createModel(vocabSize) {
    const model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 100,
        returnSequences: true,
        inputShape: [null, vocabSize],
        kernelInitializer: 'glorotUniform'
    }));
    model.add(tf.layers.lstm({
        units: 100,
        kernelInitializer: 'glorotUniform'
    }));
    model.add(tf.layers.dense({
        units: vocabSize,
        activation: 'softmax',
        kernelInitializer: 'glorotUniform'
    }));
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'categoricalCrossentropy'
    });
    return model;
}

async function prepareData(seqLength) {
    const data = await loadData();
    tokenIndex = createTokenIndex(data);
    const sequences = createSequences(data, seqLength);
    const vocabSize = Object.keys(tokenIndex).length;

    const X = [];
    const y = [];
    sequences.forEach(seq => {
        const inputSeq = seq.slice(0, seqLength).map(token => tokenIndex[token]);
        const outputToken = tokenIndex[seq[seqLength]];
        X.push(tf.oneHot(inputSeq, vocabSize));
        y.push(tf.oneHot(outputToken, vocabSize));
    });

    return {
        X: tf.stack(X),
        y: tf.stack(y),
        vocabSize
    };
}

async function trainModel(model, X, y) {
    await model.fit(X, y, {
        epochs: 50,
        batchSize: 32
    });
}

function predictNextWord(model, inputSeq, tokenIndex) {
    try {
        const inputTokens = inputSeq.map(token => {
            if (!(token in tokenIndex)) {
                throw new Error(`Token "${token}" nicht im Token-Index gefunden.`);
            }
            return tokenIndex[token];
        });
        const inputTensor = tf.oneHot(inputTokens, Object.keys(tokenIndex).length).expandDims(0);
        const prediction = model.predict(inputTensor);
        const predictedIndex = prediction.argMax(-1).dataSync()[0];
        return Object.keys(tokenIndex).find(key => tokenIndex[key] === predictedIndex);
    } catch (error) {
        console.error('Fehler bei der Vorhersage:', error);
        const fallbackWords = ['habe', 'eine', 'ist', 'wunderbare', 'ganze', 'Welt', 'erhält', 'und',
        'sich', 'in', 'ruht', 'meiner', 'trägt', 'Freund', 'ich', 'allein'];
        const randomIndex = Math.floor(Math.random() * fallbackWords.length);
        return fallbackWords[randomIndex];
    }
}

async function saveModel(model) {
    await model.save('localstorage://my-model');
}

async function loadModel() {
    return await tf.loadLayersModel('localstorage://my-model');
}

(async function () {
    const modelExists = await tf.io.listModels().then(models => 'localstorage://my-model' in models);

    if (modelExists) {
        model = await loadModel();
        console.log('Modell aus dem lokalen Speicher geladen.');
    } else {
        const {X, y, vocabSize} = await prepareData(seqLength);
        model = createModel(vocabSize);
        await trainModel(model, X, y);
        await saveModel(model);
        console.log('Modell trainiert und gespeichert.');
    }

    document.getElementById('predictButton').addEventListener('click', () => {
        const inputText = document.getElementById('inputText').value.split(' ');
        const nextWord = predictNextWord(model, inputText.slice(-seqLength), tokenIndex);
        updatePredictions(nextWord);
    });

    document.getElementById('acceptButton').addEventListener('click', () => {
        const inputText = document.getElementById('inputText').value.split(' ');
        const nextWord = predictNextWord(model, inputText.slice(-seqLength), tokenIndex);
        document.getElementById('inputText').value += ' ' + nextWord;
        updatePredictions(nextWord);
    });

    document.getElementById('autoButton').addEventListener('click', async () => {
        for (let i = 0; i < 10; i++) {
            const inputText = document.getElementById('inputText').value.split(' ');
            const nextWord = predictNextWord(model, inputText.slice(-seqLength), tokenIndex);
            document.getElementById('inputText').value += ' ' + nextWord;
            updatePredictions(nextWord);
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    });

    document.getElementById('stopButton').addEventListener('click', () => {
        // Implementiere die Logik zum Stoppen der automatischen Vorhersage
    });

    document.getElementById('resetButton').addEventListener('click', () => {
        document.getElementById('inputText').value = '';
        document.getElementById('predictions').innerText = '';
    });

    function updatePredictions(nextWord) {
        const predictionsDiv = document.getElementById('predictions');
        predictionsDiv.innerText = `Nächstes Wort: ${nextWord}`;
    }
})();
