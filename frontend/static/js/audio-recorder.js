class AudioRecorder {
    constructor(visualizerId) {
        this.visualizer = document.querySelector(visualizerId);
        this.ctx = this.visualizer.getContext('2d');
        this.animationFrame = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.isVisualizerActive = false;
        this.timer = null;
        this.seconds = 0;
        this.setupCanvas();
        this.gradientColors = {
            start: '#4050da',  // matches your primary color
            end: '#3f6cce'     // matches your secondary color
        };
        this.setupEventListeners();
    }

    startTimer() {
        if (!this.timer) {
            const timerDisplay = document.getElementById('timer');
            this.timer = setInterval(() => {
                this.seconds++;
                timerDisplay.textContent = new Date(this.seconds * 1000)
                    .toISOString().substr(14, 5);
            }, 1000);
        }
    }

    stopTimer() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
            this.seconds = 0;
            document.getElementById('timer').textContent = '00:00';
        }
    }

    startVisualization() {
        if (!this.isVisualizerActive) {
            this.setupVisualization();
            this.isVisualizerActive = true;
        }
    }

    stopVisualization() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
        // Clear canvas with fade out
        const fadeOut = () => {
            if (this.ctx.globalAlpha > 0.1) {
                this.ctx.globalAlpha -= 0.1;
                this.ctx.fillStyle = 'var(--bg-darker)';
                this.ctx.fillRect(0, -this.visualizer.height / 2, this.visualizer.width, this.visualizer.height);
                requestAnimationFrame(fadeOut);
            } else {
                this.ctx.globalAlpha = 1;
                this.ctx.fillStyle = 'var(--bg-darker)';
                this.ctx.fillRect(0, -this.visualizer.height / 2, this.visualizer.width, this.visualizer.height);
            }
        };
        fadeOut();
        this.isVisualizerActive = false;
    }

    async start() {
        try {
            if (!this.audioContext) {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.audioContext = new AudioContext();
                this.analyser = this.audioContext.createAnalyser();
                const source = this.audioContext.createMediaStreamSource(stream);
                source.connect(this.analyser);
                this.mediaRecorder = new MediaRecorder(stream);
            }
            
            this.startTimer();
            this.startVisualization();
            this.mediaRecorder.start();
            return true;
        } catch (err) {
            console.error('Error accessing microphone:', err);
            return false;
        }
    }

    stop() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.stopVisualization();
            // Don't stop the timer
        }
    }

    reset() {
        this.stopVisualization();
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
            this.seconds = 0;
            document.getElementById('timer').textContent = '00:00';
        }
    }

    setupCanvas() {
        // Set canvas size with high DPI support
        const dpr = window.devicePixelRatio || 1;
        const rect = this.visualizer.getBoundingClientRect();
        this.visualizer.width = rect.width * dpr;
        this.visualizer.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.ctx.translate(0, this.visualizer.height / 2);
    }

    setupEventListeners() {
        const stopButton = document.getElementById('stopRecording');
        const startButton = document.getElementById('startRecording');

        if (startButton) {
            startButton.addEventListener('click', async () => {
                await this.start();
            });
        }

        if (stopButton) {
            stopButton.addEventListener('click', () => {
                this.stop();
            });
        }
    }

    setupVisualization() {
        if (!this.analyser) return;
        
        this.analyser.fftSize = 512;
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const gradient = this.ctx.createLinearGradient(0, -50, 0, 50);
        gradient.addColorStop(0, this.gradientColors.start);
        gradient.addColorStop(1, this.gradientColors.end);

        const draw = () => {
            this.animationFrame = requestAnimationFrame(draw);
            this.analyser.getByteTimeDomainData(dataArray);
            
            // Clear canvas with fade effect
            this.ctx.fillStyle = 'rgba(13, 17, 23, 0.3)'; // Matches your dark background
            this.ctx.fillRect(0, -this.visualizer.height / 2, this.visualizer.width, this.visualizer.height);
            
            this.ctx.beginPath();
            this.ctx.lineWidth = 2;
            this.ctx.strokeStyle = gradient;
            
            const sliceWidth = this.visualizer.width / bufferLength;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i] / 128.0;
                const y = v * this.visualizer.height / 4;
                
                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
                
                x += sliceWidth;
            }
            
            // Mirror effect
            for (let i = bufferLength - 1; i >= 0; i--) {
                const v = dataArray[i] / 128.0;
                const y = -v * this.visualizer.height / 4;
                this.ctx.lineTo(x, y);
                x -= sliceWidth;
            }
            
            this.ctx.closePath();
            this.ctx.stroke();
            this.ctx.fillStyle = 'rgba(64, 80, 218, 0.1)'; // Slight glow effect
            this.ctx.fill();
        };
        
        draw();
    }
}