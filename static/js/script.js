document.getElementById('predictButton').addEventListener('click', () => {
    const text = document.getElementById('inputText').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: text}),
    })
        .then(response => response.json())
        .then(data => {
            const predictionsDiv = document.getElementById('predictions');
            predictionsDiv.innerHTML = '';
            data.predictions.forEach(prediction => {
                const button = document.createElement('button');
                button.textContent = prediction;
                button.addEventListener('click', () => {
                    document.getElementById('inputText').value += ' ' + prediction;
                });
                predictionsDiv.appendChild(button);
            });
        })
        .catch(error => console.error('Error:', error));
});

document.getElementById('acceptButton').addEventListener('click', () => {
    const text = document.getElementById('inputText').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: text}),
    })
        .then(response => response.json())
        .then(data => {
            const nextWord = data.predictions[0];
            document.getElementById('inputText').value += ' ' + nextWord;
        })
        .catch(error => console.error('Error:', error));
});

let autoPredictInterval;

document.getElementById('autoButton').addEventListener('click', () => {
    autoPredictInterval = setInterval(() => {
        const text = document.getElementById('inputText').value;
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({text: text}),
        })
            .then(response => response.json())
            .then(data => {
                const nextWord = data.predictions[0];
                document.getElementById('inputText').value += ' ' + nextWord;
            })
            .catch(error => console.error('Error:', error));
    }, 1000); // Vorhersage alle 1 Sekunde
});

document.getElementById('stopButton').addEventListener('click', () => {
    clearInterval(autoPredictInterval);
});

document.getElementById('resetButton').addEventListener('click', () => {
    document.getElementById('inputText').value = '';
    document.getElementById('predictions').innerHTML = '';
});
