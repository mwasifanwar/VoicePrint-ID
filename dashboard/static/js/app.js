// dashboard/static/js/app.js
class VoicePrintApp {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.recording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        
        this.initializeEventListeners();
        this.initializeCharts();
    }
    
    initializeEventListeners() {
        document.getElementById('analysisForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeAudio();
        });
        
        document.getElementById('registrationForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.registerSpeaker();
        });
        
        document.getElementById('startRecording').addEventListener('click', () => {
            this.startRecording();
        });
        
        document.getElementById('stopRecording').addEventListener('click', () => {
            this.stopRecording();
        });
    }
    
    initializeCharts() {
        const ctx = document.getElementById('statsChart').getContext('2d');
        this.statsChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Speaker Recognition', 'Emotion Detection', 'Language Detection', 'Spoof Detection'],
                datasets: [{
                    data: [85, 78, 92, 88],
                    backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Accuracy (%)'
                    }
                }
            }
        });
    }
    
    async analyzeAudio() {
        const formData = new FormData(document.getElementById('analysisForm'));
        
        try {
            const response = await fetch('/api/analyze_audio', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayResults(result.results);
            } else {
                this.showError('Analysis failed: ' + result.error);
            }
        } catch (error) {
            this.showError('Network error: ' + error.message);
        }
    }
    
    async registerSpeaker() {
        const formData = new FormData(document.getElementById('registrationForm'));
        
        try {
            const response = await fetch('/api/register_speaker', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            const resultDiv = document.getElementById('registrationResult');
            resultDiv.style.display = 'block';
            
            if (result.success) {
                resultDiv.innerHTML = `
                    <div style="color: #28a745;">
                        <strong>Success:</strong> ${result.message}
                    </div>
                `;
            } else {
                resultDiv.innerHTML = `
                    <div style="color: #dc3545;">
                        <strong>Error:</strong> ${result.error || result.message}
                    </div>
                `;
            }
        } catch (error) {
            this.showError('Registration failed: ' + error.message);
        }
    }
    
    displayResults(results) {
        const resultsDiv = document.getElementById('results');
        resultsDiv.style.display = 'block';
        
        let html = '';
        
        if (results.speaker) {
            html += `
                <div class="result-section">
                    <h4>Speaker Identification</h4>
                    <p><strong>Speaker:</strong> ${results.speaker.identified || 'Unknown'}</p>
                    <p><strong>Confidence:</strong> ${(results.speaker.confidence * 100).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${results.speaker.confidence * 100}%"></div>
                    </div>
                </div>
            `;
        }
        
        if (results.emotion) {
            html += `
                <div class="result-section">
                    <h4>Emotion Detection</h4>
                    <p><strong>Emotion:</strong> ${results.emotion.detected}</p>
                    <p><strong>Confidence:</strong> ${(results.emotion.confidence * 100).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${results.emotion.confidence * 100}%"></div>
                    </div>
                </div>
            `;
        }
        
        if (results.language) {
            html += `
                <div class="result-section">
                    <h4>Language Detection</h4>
                    <p><strong>Language:</strong> ${results.language.detected}</p>
                    <p><strong>Confidence:</strong> ${(results.language.confidence * 100).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${results.language.confidence * 100}%"></div>
                    </div>
                </div>
            `;
        }
        
        if (results.spoof_detection) {
            const spoofStatus = results.spoof_detection.is_real ? 'Real Voice' : 'Potential Spoof';
            const spoofColor = results.spoof_detection.is_real ? '#28a745' : '#dc3545';
            
            html += `
                <div class="result-section">
                    <h4>Spoof Detection</h4>
                    <p><strong>Status:</strong> <span style="color: ${spoofColor}">${spoofStatus}</span></p>
                    <p><strong>Confidence:</strong> ${(results.spoof_detection.confidence * 100).toFixed(2)}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${results.spoof_detection.confidence * 100}%"></div>
                    </div>
                </div>
            `;
        }
        
        resultsDiv.innerHTML = html;
    }
    
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.audioContext = new AudioContext();
            this.analyser = this.audioContext.createAnalyser();
            const source = this.audioContext.createMediaStreamSource(stream);
            source.connect(this.analyser);
            
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];
            
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecordedAudio();
            };
            
            this.mediaRecorder.start();
            this.recording = true;
            
            document.getElementById('startRecording').disabled = true;
            document.getElementById('stopRecording').disabled = false;
            
            this.startVisualization();
            
        } catch (error) {
            this.showError('Microphone access denied: ' + error.message);
        }
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.recording) {
            this.mediaRecorder.stop();
            this.recording = false;
            
            document.getElementById('startRecording').disabled = false;
            document.getElementById('stopRecording').disabled = true;
            
            if (this.audioContext) {
                this.audioContext.close();
            }
        }
    }
    
    startVisualization() {
        const canvas = document.getElementById('waveform');
        const ctx = canvas.getContext('2d');
        
        const draw = () => {
            if (!this.recording || !this.analyser) {
                return;
            }
            
            requestAnimationFrame(draw);
            
            const bufferLength = this.analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            this.analyser.getByteTimeDomainData(dataArray);
            
            ctx.fillStyle = '#2c3e50';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#00ff00';
            ctx.beginPath();
            
            const sliceWidth = canvas.width * 1.0 / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * canvas.height / 2;
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.stroke();
        };
        
        draw();
    }
    
    async processRecordedAudio() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        try {
            const response = await fetch('/api/analyze_audio', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.displayRealTimeResults(result.results);
            }
        } catch (error) {
            this.showError('Real-time analysis failed: ' + error.message);
        }
    }
    
    displayRealTimeResults(results) {
        const resultsDiv = document.getElementById('realTimeResults');
        
        let html = '<div class="result-box"><h4>Real-time Analysis</h4>';
        
        if (results.emotion) {
            html += `<p><span class="real-time-indicator"></span> Emotion: ${results.emotion.detected} (${(results.emotion.confidence * 100).toFixed(1)}%)</p>`;
        }
        
        if (results.language) {
            html += `<p><span class="real-time-indicator"></span> Language: ${results.language.detected} (${(results.language.confidence * 100).toFixed(1)}%)</p>`;
        }
        
        if (results.spoof_detection) {
            const spoofText = results.spoof_detection.is_real ? 'Real' : 'Spoof';
            html += `<p><span class="real-time-indicator"></span> Authenticity: ${spoofText} (${(results.spoof_detection.confidence * 100).toFixed(1)}%)</p>`;
        }
        
        html += '</div>';
        resultsDiv.innerHTML = html;
    }
    
    showError(message) {
        alert('Error: ' + message);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new VoicePrintApp();
});