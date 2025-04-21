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
import uuid
import json  # Ensure this is imported at the top of the file

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.interview_project.async_file import run_async
from backend.interview_project.interview_flow import *
from backend.interview_project.audio_processing import save_audio_to_wav_async
from flask_socketio import SocketIO
from flask import Flask

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['INTERVIEW_INSTANCES'] = {}
# Store interview instances and their generators
interview_generators = {}


UPLOAD_FOLDER = 'uploads/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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


@app.route('/job-details', methods=['GET', 'POST'])
@login_required
def job_details():
    if request.method == 'POST':
        # Get form data
        job_title = request.form.get('job_title')
        experience = request.form.get('experience_text')
        # Store job title and experience in session
        session['job_title'] = job_title
        session['experience'] = experience
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
    from backend.main import main as start_interview
    interview_instance, job_title, experience = run_async(
        start_interview(job_title=job_title, experience_text=experience_text)
    )
    print("interview_instance created from interview_start_page")
    session['interview_instance_id'] = id(interview_instance)  # Store the instance ID in the session
    app.config['INTERVIEW_INSTANCES'] = app.config.get('INTERVIEW_INSTANCES', {})
    app.config['INTERVIEW_INSTANCES'][id(interview_instance)] = interview_instance  # Store the instance globally
    session["job_title"] = job_title
    session["experience"] = experience
    session["current_question_index"] = 0  # Reset question index
    # Set the INTERVIEW_INSTANCE key
    app.config['INTERVIEW_INSTANCE'] = interview_instance

    print(f"{interview_instance} in app.py")
    return render_template(
        "interview_start_page.html",
        job_title=job_title,
        experience_text=experience,
        interview_instance_id=session['interview_instance_id'],
    )


@app.route('/going_to_conduct_interview', methods=['POST', 'GET'])
@login_required
def going_to_conduct_interview():
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

        # Fetch the first question
        question = get_next_question(interview_instance_id)
        question_data = question.get_json()
        print(question)
    
        try:
            # Access the data
            question_text = question_data['question']
            question_audio = question_data['audio']
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing question data: {e}", file=sys.stderr)
            flash('Invalid question data received.', 'error')
            return redirect(url_for('interview_start_page'))

        print(f"question_text{question_text},question_audio{question_audio}")
        return render_template(
            "going_to_conduct_interview.html",
            question_text=question_text,
            question_audio=question_audio,
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


def get_next_question(interview_instance_id):
    # Retrieve the interview instance from global storage
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

    if not interview_instance:
        return jsonify({'status': 'error', 'message': 'Interview instance not found.'}), 404

    try:
        # Get the next question from the generator
        question_data = run_async(interview_instance.conduct_interview())
        print(f"Question Data: {question_data},question text{question_data["text"]}inside get generate question")  # Debugging line
        return jsonify({
            'status': 'success',
            'question': question_data['text'],
            'audio': question_data['audio']
        })
    except StopIteration:
        # No more questions available
        return jsonify({
            'status': 'completed',
            'message': 'The interview is complete. Thank you for participating!'
        })
    except Exception as e:
        print(f"Error fetching next question: {e}", file=sys.stderr)
        return jsonify({'status': 'error', 'message': 'An error occurred while fetching the question.'}), 500


# WebSocket event for fetching the next question
@socketio.on('fetch_next_question')
async def handle_fetch_next_question(data):
    """
    Handles the 'fetch_next_question' event from the frontend.
    Dynamically generates and emits the next question in real-time.
    """
    try:
        # Retrieve the interview instance from the session
        interview_instance_id = session.get('interview_instance_id')
        interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

        if not interview_instance:
            emit('next_question', {'status': 'error', 'message': 'Interview instance not found.'})
            return

        # Dynamically generate the next question
        question_data = await interview_instance.generate_next_question()
        if question_data:
            # Emit the question to the frontend in real-time
            emit('next_question', {
                'status': 'success',
                'question': question_data['text'],
                'audio': question_data['audio']
            })
        else:
            # Emit a completion message if no more questions are available
            emit('next_question', {'status': 'completed', 'message': 'The interview is complete.'})
    except Exception as e:
        print(f"Error in handle_fetch_next_question: {e}", file=sys.stderr)
        emit('next_question', {'status': 'error', 'message': str(e)})

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
        emit('question_response', {'status': 'error', 'message': str(e)})

@socketio.on('submit_audio_answer')
async def handle_audio_answer(data):
    """
    Handles the audio answer submitted by the frontend.

    Args:
        data (dict): Contains the audio file path and other metadata.
    """
    try:
        audio_file_path = data.get('audio_file_path')
        interview_instance_id = session.get('interview_instance_id')
        interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)

        if not interview_instance:
            emit('audio_submission', {'status': 'error', 'message': 'Interview instance not found.'})
            return

        # Check if the audio file exists
        if not audio_file_path or not os.path.exists(audio_file_path):
            emit('audio_submission', {'status': 'error', 'message': 'Audio file not found.'})
            return

        # Process the audio response
        user_response, feedback = await interview_instance.process_and_record_answer(audio_file_path)

        if user_response:
            emit('audio_submission', {
                'status': 'success',
                'user_response': user_response,
                'feedback': feedback
            })
        else:
            emit('audio_submission', {
                'status': 'error',
                'message': feedback
            })
    except Exception as e:
        print(f"Error handling audio answer: {e}", file=sys.stderr)
        emit('audio_submission', {'status': 'error', 'message': 'An error occurred while processing the audio.'})

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file uploaded.'}), 400

    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(file_path)
    return jsonify({'status': 'success', 'file_path': file_path}), 200

# Initialize the interview process or fetch the first question
@app.route('/live_interview', methods=['POST', 'GET'])
@login_required
def live_interview():
    if 'user' not in session:
        flash('Please login to access interviews', 'warning')
        return redirect(url_for('login'))
    if request.method == 'POST':
        # Retrieve the interview instance from global storage
        interview_instance_id = session.get('interview_instance_id')
        interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)
        if not interview_instance:
            flash('Interview instance not found. Please restart the interview.', 'error')
            return redirect(url_for('interview_start_page'))
        # Use the interview instance to fetch the first question
        # Initialize the interview instance if not already done
        if interview_instance_id not in app.config['INTERVIEW_INSTANCES']:
            flash('Interview instance not found. Please restart the interview.', 'error')
            return redirect(url_for('interview_start_page'))
        # Fetch the first question
        try:
            question_data =  asyncio.run(interview_instance.conduct_interview())
            if question_data == "completed":
                flash('The interview is complete. Thank you for participating!', 'success')
                return redirect(url_for('show_result'))
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
@app.route('/show_result', methods=['POST', 'GET'])
@login_required
def show_result():
    if 'user' not in session:
        flash('Please login to access interviews', 'warning')
        return redirect(url_for('login'))
    interview_instance_id = session.get('interview_instance_id')
    interview_instance = app.config['INTERVIEW_INSTANCES'].get(interview_instance_id)
    if not interview_instance:
        flash('Interview instance not found. Please restart the interview.', 'error')
        return redirect(url_for('interview_start_page'))
    # Use the interview instance to fetch the results
    try:
        result_data = asyncio.run(interview_instance.get_results())
        print(f"Result Data: {result_data}")  # Debugging line
    except Exception as e:
        print(f"Error fetching results: {e}", file=sys.stderr)
        result_data = "No results available at the moment."

    return render_template(
        "show_result.html",
        result_data=result_data,
    )
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)