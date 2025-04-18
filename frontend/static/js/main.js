document.addEventListener('DOMContentLoaded', function() {
    const startButton = document.getElementById('startRecording');
    const stopButton = document.getElementById('stopRecording');
    const visualizer = document.getElementById('visualizer');
    const timerDisplay = document.getElementById('timer');
    
    let mediaRecorder;
    let audioContext;
    let analyser;
    let dataArray;
    let timer;
    let seconds = 0;

    // Initialize audio context
    async function initAudio() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new AudioContext();
            analyser = audioContext.createAnalyser();
            const source = audioContext.createMediaStreamSource(stream);
            source.connect(analyser);
            
            mediaRecorder = new MediaRecorder(stream);
            setupVisualization();
            
            return true;
        } catch (err) {
            console.error('Error accessing microphone:', err);
            return false;
        }
    }

    // Set up audio visualization
    function setupVisualization() {
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        
        const ctx = visualizer.getContext('2d');
        const width = visualizer.width;
        const height = visualizer.height;
        
        function draw() {
            requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);
            
            ctx.fillStyle = 'rgb(20, 20, 20)';
            ctx.fillRect(0, 0, width, height);
            
            const barWidth = (width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for(let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i]/2;
                ctx.fillStyle = `rgb(${barHeight + 100},50,50)`;
                ctx.fillRect(x, height-barHeight/2, barWidth, barHeight);
                x += barWidth + 1;
            }
        }
        
        draw();
    }

    // Event Listeners
    startButton.addEventListener('click', async () => {
        if (!mediaRecorder) {
            const initialized = await initAudio();
            if (!initialized) return;
        }
        
        mediaRecorder.start();
        startButton.disabled = true;
        stopButton.disabled = false;
        
        // Start timer
        timer = setInterval(() => {
            seconds++;
            timerDisplay.textContent = new Date(seconds * 1000)
                .toISOString().substr(14, 5);
        }, 1000);
    });

    stopButton.addEventListener('click', () => {
        mediaRecorder.stop();
        clearInterval(timer);
        seconds = 0;
        timerDisplay.textContent = '00:00';
        
        startButton.disabled = false;
        stopButton.disabled = true;
    });
});