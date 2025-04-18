import sys
import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from functools import wraps
from datetime import datetime
import asyncio
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.interview_project.async_file import run_async
from backend.main import main as start_interview
from backend.interview_project.interview_flow import *
from backend.interview_project.async_file import run_async
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

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

@app.route('/live-interview', methods=["POST", "GET"])
@login_required
def live_interview():
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
    #global interviewer_instance
    interview_instance,job_title,experience_text = run_async(start_interview(job_title=job_title, experience_text=experience_text))
    interview_instance = interview_instance
    print(f"{interview_instance} in app.py")
    # Fetch the first question from the interview process
    try:
        question_data = asyncio.run(interview_instance.conduct_interview().__anext__())
        first_question = question_data["text"]
        question_audio = question_data["audio"]
        print(f"first_question: {first_question} question_audio: {question_audio}")
    except StopIteration:
        flash('No questions available for the interview.', 'error')
        return redirect(url_for('job_details'))
    return render_template(
        "live_interview.html",
        job_title=job_title,
        experience_text=experience_text,
        first_question=first_question,
        question_audio=question_audio,
    )
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
        return redirect(url_for('live_interview'))
    return render_template('job_details.html')


@app.route('/results')
@login_required
def show_result():
    """global interviewer_instance
    
    if not interviewer_instance:
        return "Interview not started.", 400

    # Example: fill these before evaluation
    interviewer_instance.asked_questions = [
        "What is the difference between a list and a tuple in Python?",
        "Explain how a hash table works.",
        "What is the time complexity of binary search?",
    ]

    interviewer_instance.answers = [
        "A list is mutable and can be changed, whereas a tuple is immutable and cannot be modified after creation.",
        "A hash table uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.",
        "The time complexity of binary search is O(log n).",
    ]"""
    mock_results = {
        'confidence': '75%',
        'clarity': 'Good',
        'response_time': '45 seconds',
        'tip': 'Try to provide more detailed examples in your responses'
    }
    #result = interviewer_instance.generate_and_save_feedback()
    return render_template('results.html', mock_results=mock_results)
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
    # Here you would process the audio file and generate feedback
    return jsonify({
        'status': 'success',
        'feedback': 'Good response! Consider providing more specific examples.',
        'confidence_score': 85
    })

@app.route('/api/interview/next-question', methods=['POST'])
@login_required
def next_question():
    # Get the current question index from the session
    current_index = session.get('current_question_index', 0)
    
    # Check if there are more questions
    if current_index < len(mock_questions):
        question = mock_questions[current_index]
        session['current_question_index'] = current_index + 1
        return render_template('live_interview.html', question=question)  # Render the next question
    else:
        # If all questions are completed, redirect to results.html
        flash('Interview completed!', 'success')
        return redirect(url_for('show_result'))  # Redirect to results.html

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)