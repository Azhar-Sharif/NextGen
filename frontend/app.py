import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from datetime import datetime
import asyncio
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64
import wave
import numpy as np
import subprocess

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.interview_project.async_file import run_async
from backend.main import main as start_interview
from backend.interview_project.interview_flow import *
from backend.interview_project.async_file import run_async
from backend.interview_project.audio_processing import save_audio_to_wav_async

# Initialize Flask and Flask-SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['INTERVIEW_INSTANCES'] = {}
socketio = SocketIO(app)

# Mock data for demo
mock_interview_data = {
    'completed_interviews': 5,
    'average_score': '85%',
    'upcoming_interviews': 2,
    'improvement_tips': [
        'Work on maintaining consistent eye contact',
        'Practice speaking at a slower pace',
        'Provide more specific examples in answers'
    ]
}

mock_results = {
    'confidence': '75%',
    'clarity': 'Good',
    'response_time': '45 seconds',
    'tip': 'Try to provide more detailed examples in your responses'
}

# Mock questions for the interview
mock_questions = [
    "Tell me about yourself.",
    "What are your strengths and weaknesses?",
    "Describe a challenging situation and how you handled it.",
    "Why do you want to work for this company?",
    "Where do you see yourself in 5 years?"
]

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/home')
def landing():
    return render_template('landing.html')

@app.route('/live-interview', methods=['POST', 'GET'])
@login_required
def live_interview():
    if request.method == 'POST':
        # Retrieve form data
        job_title = request.form.get('job_title')
        experience_text = request.form.get('experience_text')
        interview_instance_id = session.get('interview_instance_id')

        # Retrieve the interview instance from global storage
        interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

        if not interview_instance:
            flash('Interview instance not found. Please restart the interview.', 'error')
            return redirect(url_for('interview_start_page'))

        # Use these values to conduct the interview
        print(f"Job Title: {job_title}, Experience: {experience_text}, Instance: {interview_instance}")

        # Initialize the interview process or fetch the first question
        try:
            question_data = asyncio.run(interview_instance.conduct_interview().__anext__())
            if question_data == "completed":
                flash('The interview is complete. Thank you for participating!', 'success')
                return redirect(url_for('show_results'))
            first_question = question_data["text"]
            question_audio = question_data["audio"]
            print(f"First Question: {first_question}, Audio: {question_audio}")
        except StopIteration:
            flash('No questions available for the interview.', 'error')
            return redirect(url_for('job_details'))
        except Exception as e:
            print(f"Error fetching the first question: {e}", file=sys.stderr)
            first_question = "No questions available at the moment."
            question_audio = ""

        return render_template(
            "live_interview.html",
            first_question=first_question,
            question_audio=question_audio,
        )
    else:
        flash('Invalid request method.', 'error')
        return redirect(url_for('interview_start_page'))

@app.route('/job-details', methods=['GET', 'POST'])
@login_required
def job_details():
    if request.method == 'POST':
        # Get form data
        job_title = request.form.get('job_title')
        experience = request.form.get('experience_text')
        
        if not job_title or not experience:
            flash('Please fill in all fields', 'error')
            return redirect(url_for('job_details'))
        return redirect(url_for('interview_start_page'))
    return render_template('job_details.html')

@app.route('/interview_start_page', methods=["POST", "GET"])
@login_required
def interview_start_page():
    if 'user' not in session:
        flash('Please login to access interviews', 'warning')
        return redirect(url_for('login'))

    if request.method == 'POST':
        job_title = request.form.get('job_title')
        experience_text = request.form.get('experience_text')
        session['job_title'] = job_title
        session['experience'] = experience_text
        session['current_question_index'] = 0  # Reset question index
    else:
        job_title = session.get('job_title')
        experience_text = session.get('experience')

    if not job_title or not experience_text:
        flash('Missing job details.', 'error')
        return redirect(url_for('job_details'))

    # Initialize the Interviewer instance
    interview_instance, job_title, experience = run_async(
        start_interview(job_title=job_title, experience_text=experience_text)
    )
    session['interview_instance_id'] = id(interview_instance)  # Store the instance ID in the session
    app.config['INTERVIEW_INSTANCES'] = app.config.get('INTERVIEW_INSTANCES', {})
    app.config['INTERVIEW_INSTANCES'][id(interview_instance)] = interview_instance  # Store the instance globally

    # Set the INTERVIEW_INSTANCE key
    app.config['INTERVIEW_INSTANCE'] = interview_instance

    print(f"{interview_instance} in app.py")
    return render_template(
        "interview_start_page.html",
        job_title=job_title,
        experience_text=experience,
        interview_instance_id=session['interview_instance_id'],
    )

@app.route('/results')
@login_required
def show_result():
    # Get the next question and audio from query parameters
    next_question = request.args.get('question', None)
    question_audio = request.args.get('audio', None)

    # If no question is provided, show mock results
    if not next_question:
        mock_results = {
            'confidence': '75%',
            'clarity': 'Good',
            'response_time': '45 seconds',
            'tip': 'Try to provide more detailed examples in your responses'
        }
        return render_template('results.html', mock_results=mock_results)

    # Render the results page with the next question
    return render_template(
        'results.html',
        next_question=next_question,
        question_audio=question_audio
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('job_details'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if email and password:
            session['user'] = {'email': email}
            flash('Successfully logged in!', 'success')
            return redirect(url_for('landing'))  # Changed from dashboard to live_interview
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        
        if not all([name, email, password1, password2]):
            flash('All fields are required!', 'danger')
        elif password1 != password2:
            flash('Passwords do not match!', 'danger')
        else:
            flash('Account created successfully!', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Successfully logged out!', 'success')
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', data=mock_interview_data)

@app.route('/api/interview/complete', methods=['POST'])
@login_required
def complete_interview():
    # Process the interview completion
    # Add your interview processing logic here
    return jsonify({
        'status': 'success',
        'redirect': url_for('results')
    })

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/reset_password')
def reset_password():
    return render_template('reset_password.html')

@app.route('/demo')
def demo_interview():
    mock_demo_data = {
        'question': 'Tell me about yourself.',
        'tips': [
            'Maintain good eye contact',
            'Speak clearly and confidently',
            'Structure your answer with past, present, and future'
        ],
        'time_limit': '2 minutes'
    }
    return render_template('demo.html', data=mock_demo_data)

@app.route('/api/interview/submit', methods=['POST'])
@login_required
def submit_answer():
    try:
        if 'audio' not in request.files:
            return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400

        audio_file = request.files['audio']
        print(f"Received audio file: {audio_file.filename}")  # Debugging log

        # Process the audio file (add your logic here)
        # Example: Save the file temporarily
        temp_audio_path = os.path.join('temp', audio_file.filename)
        os.makedirs('temp', exist_ok=True)
        audio_file.save(temp_audio_path)

        # Simulate processing and return success
        return jsonify({'status': 'success', 'message': 'Audio processed successfully.'})
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr)
        return jsonify({'status': 'error', 'message': 'An error occurred while processing the audio.'}), 500

@app.route('/api/interview/next-question', methods=['POST'])
@login_required
def get_next_question():
    # Retrieve the interview instance from global storage
    interview_instance_id = session.get('interview_instance_id')
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

    if not interview_instance:
        return jsonify({'status': 'error', 'message': 'Interview instance not found.'}), 404

    try:
        # Get the next question from the generator
        question_data = asyncio.run(interview_instance.conduct_interview().__anext__())
        print(f"Question Data: {question_data}")  # Debugging line
        next_question = question_data["text"]
        question_audio = question_data["audio"]

        return jsonify({
            'status': 'success',
            'question': next_question,
            'audio': question_audio
        })
    except StopIteration:
        # No more questions available
        return jsonify({
            'status': 'completed',
            'message': 'The interview is complete. Thank you for participating!'
        })
    except Exception as e:
        print(f"Error fetching next question: {e}", file=sys.stderr)
        return jsonify({'status': 'error', 'message': 'An error occurred while fetching the next question.'}), 500

# Store interview instances and their generators
interview_generators = {}

# WebSocket event for fetching the next question
@socketio.on('fetch_next_question')
def handle_fetch_next_question(data):
    """Handles the 'fetch_next_question' event and sends the next question to the frontend."""
    interview_instance_id = session.get('interview_instance_id')
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

    if not interview_instance:
        emit('next_question', {'status': 'error', 'message': 'Interview instance not found.'})
        return

    # Retrieve or initialize the generator
    if interview_instance_id not in interview_generators:
        interview_generators[interview_instance_id] = interview_instance.conduct_interview()

    generator = interview_generators[interview_instance_id]

    try:
        # Fetch the next question from the generator
        question_data = asyncio.run(generator.__anext__())
        emit('next_question', {
            'status': 'success',
            'question': question_data["text"],
            'audio': question_data["audio"]
        })
    except StopIteration:
        # Remove the generator when the interview is complete
        del interview_generators[interview_instance_id]
        emit('next_question', {'status': 'completed', 'message': 'The interview is complete.'})
    except Exception as e:
        print(f"Error fetching next question: {e}", file=sys.stderr)
        emit('next_question', {'status': 'error', 'message': 'An error occurred while fetching the next question.'})

@socketio.on('ask_question')
def handle_ask_question(data):
    """Handles the 'ask_question' event and sends the question to the frontend."""
    interview_instance_id = session.get('interview_instance_id')
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

    if not interview_instance:
        emit('question_response', {'status': 'error', 'message': 'Interview instance not found.'})
        return

    try:
        # Fetch the next question from the interview instance
        question = data.get('question')  # Ensure this is set correctly
        if not question:
            question = "Default question text"  # Fallback if no question is provided

        question_text, question_audio = asyncio.run(interview_instance.ask_question(question))
        emit('question_response', {
            'status': 'success',
            'question': question_text,
            'audio': question_audio
        })
    except Exception as e:
        print(f"Error in ask_question: {e}", file=sys.stderr)
        emit('question_response', {'status': 'error', 'message': 'An error occurred while asking the question.'})

@socketio.on('submit_audio_answer')
def handle_audio_answer(data):
    """Handles the 'submit_audio_answer' event and processes the audio answer."""
    interview_instance_id = session.get('interview_instance_id')
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

    if not interview_instance:
        emit('answer_feedback', {'status': 'error', 'message': 'Interview instance not found.'})
        return

    try:
        # Save the audio file
        audio_binary = data.get('audio')
        if not audio_binary:
            emit('answer_feedback', {'status': 'error', 'message': 'No audio data received.'})
            return

        audio_file_path = os.path.join('frontend', 'static', 'audio', 'response.wav')
        with open(audio_file_path, 'wb') as f:
            f.write(audio_binary)

        # Process the audio file
        result = asyncio.run(interview_instance.record_and_process_answer(audio_file_path))
        emit('answer_feedback', result)
    except Exception as e:
        print(f"Error processing audio answer: {e}", file=sys.stderr)
        emit('answer_feedback', {'status': 'error', 'message': 'An error occurred while processing the audio answer.'})

@app.route('/submit_answer_v2', methods=['POST'])
def submit_answer_v2():
    """Handles the submission of the user's recorded answer."""
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400
    if 'interview_instance_id' not in session:
        return jsonify({'status': 'error', 'message': 'Interview instance ID not found in session.'}), 400
    interview_instance_id = session['interview_instance_id']
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)
    if not interview_instance:
        return jsonify({'status': 'error', 'message': 'Interview instance not found.'}), 404
    # Use the correct path for the temp folder
    temp_folder = os.path.join(os.path.dirname(__file__), 'temp')
    os.makedirs(temp_folder, exist_ok=True)

    audio_file = request.files['audio']
    temp_audio_path = os.path.join(temp_folder, audio_file.filename)
    converted_audio_path = os.path.join(temp_folder, 'converted_audio.wav')

    try:
        # Save the uploaded audio file temporarily
        audio_file.save(temp_audio_path)

        # Check if the file is a valid WAV file
        try:
            with wave.open(temp_audio_path, 'rb') as audio:
                pass  # File is valid WAV
        except wave.Error:
            # Convert the file to WAV format using ffmpeg
            ffmpeg_command = [
                'ffmpeg', '-i', temp_audio_path, '-ar', '16000', '-ac', '1', converted_audio_path
            ]
            subprocess.run(ffmpeg_command, check=True)
            temp_audio_path = converted_audio_path  # Use the converted file

        # Validate the audio file duration
        with wave.open(temp_audio_path, 'rb') as audio:
            frame_rate = audio.getframerate()
            num_frames = audio.getnframes()
            duration = num_frames / float(frame_rate)

            if duration < 0.1:
                os.remove(temp_audio_path)
                return jsonify({'status': 'error', 'message': 'Audio file is too short. Minimum length is 0.1 seconds.'}), 400

        # Process the audio file using the Interviewer class
        interviewer = app.config.get('INTERVIEW_INSTANCE')
        if not interviewer:
            return jsonify({'status': 'error', 'message': 'Interview instance not found.'}), 500
        print(f"Processing audio file: {temp_audio_path}")  # Debugging log
        # Use the Interviewer class to process the audio and get feedback
        
        user_response, feedback = asyncio.run(interviewer.record_and_process_answer(temp_audio_path))

        # Clean up the temporary files
        os.remove(temp_audio_path)
        if os.path.exists(converted_audio_path):
            os.remove(converted_audio_path)

        if user_response:
            return jsonify({'status': 'success', 'response': user_response, 'feedback': feedback})
        else:
            return jsonify({'status': 'error', 'message': feedback}), 500
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}", file=sys.stderr)
        return jsonify({'status': 'error', 'message': 'Failed to convert audio file to WAV format.'}), 500
    except Exception as e:
        print(f"Error processing answer: {e}", file=sys.stderr)
        return jsonify({'status': 'error', 'message': 'An error occurred while processing the answer.'}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)