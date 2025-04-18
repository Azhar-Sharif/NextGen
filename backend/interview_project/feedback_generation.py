import asyncio
import json
import os
import tempfile
import aiofiles
import sys
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from openai import AsyncOpenAI
from datetime import datetime

async def generate_overall_feedback(api_key, questions, answers, job_description, user_experience, rag_engine=None):
    """
    Generates comprehensive feedback for an interview based on questions and answers.
    
    Args:
        api_key (str): OpenAI API key
        questions (list): List of interview questions
        answers (list): List of candidate answers
        job_description (str): Job description text
        user_experience (str): Candidate's experience description
        rag_engine (RAGEngine, optional): RAG engine for technical evaluation
        
    Returns:
        str: JSON-formatted feedback
    """
    client = AsyncOpenAI(api_key=api_key)
    
    # Validate input data
    if not questions or not answers or len(questions) == 0 or len(answers) == 0:
        print("No valid interview data to generate feedback.", file=sys.stderr)
        return json.dumps(create_fallback_json("Unable to generate feedback: No valid interview data was provided."))
    
    # Process question-answer pairs with RAG evaluation for technical questions
    qa_pairs, technical_scores, rag_evaluations = await process_qa_pairs(
        questions, answers, rag_engine, job_description, client
    )
    
    # Get relevant technical content from RAG if available
    relevant_technical_content = await get_relevant_technical_content(
        rag_engine, job_description, questions
    )
    
    # Create RAG evaluation summary if available
    rag_summary = create_rag_summary(technical_scores, rag_evaluations)
    
    # Generate feedback using LLM
    feedback_json = await generate_feedback_with_llm(
        client, job_description, user_experience, qa_pairs, rag_summary, 
        relevant_technical_content, technical_scores
    )
    
    return json.dumps(feedback_json)


async def process_qa_pairs(questions, answers, rag_engine, job_description, client):
    """
    Process question-answer pairs and evaluate technical answers with RAG if available.
    
    Returns:
        tuple: (qa_pairs_text, technical_scores, rag_evaluations)
    """
    qa_pairs = []
    technical_scores = []
    rag_evaluations = []
    
    for i in range(min(len(questions), len(answers))):
        question = questions[i]
        answer = answers[i]
        
        # Check if the question is technical in nature
        question_is_technical = is_technical_question(question)
        
        # If RAG engine is available and question is technical, evaluate the answer
        if rag_engine and question_is_technical:
            try:
                # Use RAG to evaluate this specific technical answer
                evaluation = await rag_engine.evaluate_technical_answer(
                    question=question,
                    answer=answer,
                    job_description=job_description,
                    openai_client=client
                )
                
                # Store the evaluation score and feedback
                score = evaluation.get("score", 5)
                technical_scores.append(score)
                
                # Add the evaluation to our list
                rag_evaluations.append({
                    "question": question,
                    "answer": answer,
                    "score": score,
                    "feedback": evaluation.get("feedback", "")
                })
                
                # Add annotation about the RAG evaluation
                qa_pairs.append(f"Question: {question}\nAnswer: {answer}\nRAG Technical Evaluation Score: {score}/10")
            except Exception as e:
                print(f"Error using RAG to evaluate technical answer: {e}", file=sys.stderr)
                qa_pairs.append(f"Question: {question}\nAnswer: {answer}")
        else:
            qa_pairs.append(f"Question: {question}\nAnswer: {answer}")
    
    return qa_pairs, technical_scores, rag_evaluations


def is_technical_question(question):
    """
    Determines if a question is technical in nature.
    
    Args:
        question (str): The question text
        
    Returns:
        bool: True if the question is technical, False otherwise
    """
    technical_terms = [
        "algorithm", "model", "data", "statistic", "machine learning", 
        "code", "program", "neural", "regression", "classification"
    ]
    return any(term in question.lower() for term in technical_terms)


async def get_relevant_technical_content(rag_engine, job_description, questions):
    """
    Retrieves relevant technical content from RAG engine if available.
    
    Returns:
        str: Formatted technical content for evaluation
    """
    if not rag_engine:
        return ""
        
    try:
        # Create a combined query from job description and top questions
        combined_query = f"{job_description} {' '.join(questions[:3])}"
        
        # Retrieve relevant context
        retrieved_context = rag_engine.retrieve(combined_query, k=3)
        
        # Format context for the prompt
        relevant_content = "\n\nRELEVANT TECHNICAL CONTENT FOR EVALUATION:\n"
        relevant_content += "\n---\n".join(
            [f"Content: {item['chunk']['text']}" for item in retrieved_context]
        )
        return relevant_content
    except Exception as e:
        print(f"Error retrieving RAG content for feedback: {e}", file=sys.stderr)
        return ""


def create_rag_summary(technical_scores, rag_evaluations):
    """
    Creates a summary of RAG-based technical evaluations.
    
    Returns:
        str: RAG evaluation summary text
    """
    if not technical_scores:
        return ""
        
    avg_technical_score = sum(technical_scores) / len(technical_scores)
    rag_summary = f"\n\nRAG TECHNICAL EVALUATION SUMMARY:\nAverage Technical Score: {avg_technical_score:.2f}/10\n"
    
    for eval_data in rag_evaluations:
        rag_summary += f"\nQuestion: {eval_data['question']}\nScore: {eval_data['score']}/10\nFeedback: {eval_data['feedback']}\n"
    
    return rag_summary


async def generate_feedback_with_llm(client, job_description, user_experience, qa_pairs, 
                               rag_summary, relevant_technical_content, technical_scores):
    """
    Generates feedback using the LLM and blends it with RAG evaluation if available.
    
    Returns:
        dict: Feedback JSON object
    """
    qa_text = "\n\n".join(qa_pairs)
    
    # Create the prompt for generating feedback in strict JSON format
    prompt = f"""
    You are an expert interviewer for data science positions. Based on the following interview questions and answers, 
    generate comprehensive feedback for the candidate.
    
    JOB DESCRIPTION: {job_description}
    
    CANDIDATE EXPERIENCE: {user_experience}
    
    INTERVIEW TRANSCRIPT:
    {qa_text}
    {rag_summary}
    {relevant_technical_content}
    
    Please provide structured feedback in STRICTLY VALID JSON format with the following components:
    {{
      "overall_score": A number from 1-10 representing the candidate's overall performance,
      "category_scores": {{
        "Technical Knowledge": A number from 1-10,
        "Communication": A number from 1-10,
        "Problem Solving": A number from 1-10,
        "Experience": A number from 1-10
      }},
      "strengths": [3-5 strongest points as individual strings in an array],
      "areas_for_improvement": [3-5 areas to improve as individual strings in an array],
      "detailed_feedback": {{
        "technical_skills": "Feedback on technical skills",
        "communication_skills": "Feedback on communication",
        "problem_solving": "Feedback on problem solving",
        "experience_match": "Feedback on experience relevance"
      }},
      "raw_feedback": "A summary paragraph with overall assessment"
    }}
    
    YOUR RESPONSE MUST BE VALID JSON ONLY, NO OTHER TEXT, and MUST MATCH THE EXACT SCHEMA ABOVE.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are an expert interview evaluator for data science positions. You must return your response as valid JSON only, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"}  # Force JSON response format
        )
        
        feedback_text = response.choices[0].message.content.strip()
        
        # Parse and validate the JSON
        try:
            feedback_json = json.loads(feedback_text)
            
            # Blend RAG technical scores if available
            feedback_json = blend_rag_scores(feedback_json, technical_scores)
            
            # Ensure all required fields are present
            feedback_json = validate_feedback_json(feedback_json)
            
            return feedback_json
            
        except json.JSONDecodeError as e:
            print(f"Error parsing feedback JSON: {e}", file=sys.stderr)
            print(f"Raw response: {feedback_text}", file=sys.stderr)
            return create_fallback_json(feedback_text if feedback_text else "No feedback was generated")
            
    except Exception as e:
        print(f"Error generating overall feedback: {e}", file=sys.stderr)
        return create_fallback_json(f"An error occurred while generating feedback: {str(e)}")


def blend_rag_scores(feedback_json, technical_scores):
    """
    Blends RAG-based technical scores with LLM-generated scores.
    
    Returns:
        dict: Updated feedback JSON with blended scores
    """
    if not technical_scores or "category_scores" not in feedback_json or "Technical Knowledge" not in feedback_json["category_scores"]:
        return feedback_json
        
    # Calculate the average of the RAG technical scores
    avg_rag_score = sum(technical_scores) / len(technical_scores)
    
    # Get the LLM's technical score
    llm_technical_score = feedback_json["category_scores"]["Technical Knowledge"]
    
    # Blend the scores (60% RAG, 40% LLM)
    blended_score = (0.6 * avg_rag_score) + (0.4 * llm_technical_score)
    
    # Update the technical score
    feedback_json["category_scores"]["Technical Knowledge"] = round(blended_score, 1)
    
    # Recalculate overall score based on updated technical score
    all_scores = list(feedback_json["category_scores"].values())
    feedback_json["overall_score"] = round(sum(all_scores) / len(all_scores), 1)
    
    # Note the use of RAG evaluation in the raw feedback
    feedback_json["raw_feedback"] += "\n\nNote: Technical Knowledge score incorporates RAG-based evaluation."
    
    return feedback_json


def validate_feedback_json(feedback_json):
    """
    Ensures all required fields are present in the feedback JSON.
    
    Returns:
        dict: Validated and completed feedback JSON
    """
    # Ensure all required keys are present with valid values
    required_keys = ["overall_score", "category_scores", "strengths", "areas_for_improvement", "detailed_feedback", "raw_feedback"]
    for key in required_keys:
        if key not in feedback_json:
            if key in ["overall_score"]:
                feedback_json[key] = 0
            elif key in ["category_scores"]:
                feedback_json[key] = {
                    "Technical Knowledge": 0,
                    "Communication": 0, 
                    "Problem Solving": 0,
                    "Experience": 0
                }
            elif key in ["strengths", "areas_for_improvement"]:
                feedback_json[key] = ["No specific points identified"]
            elif key == "detailed_feedback":
                feedback_json[key] = {
                    "technical_skills": "Not evaluated",
                    "communication_skills": "Not evaluated",
                    "problem_solving": "Not evaluated",
                    "experience_match": "Not evaluated"
                }
            else:
                feedback_json[key] = "Not provided"
    
    # Ensure category_scores has all required categories
    required_categories = ["Technical Knowledge", "Communication", "Problem Solving", "Experience"]
    if "category_scores" in feedback_json and feedback_json["category_scores"] is not None:
        for category in required_categories:
            if category not in feedback_json["category_scores"]:
                feedback_json["category_scores"][category] = 0
    else:
        feedback_json["category_scores"] = {category: 0 for category in required_categories}
    
    # Ensure strengths and areas_for_improvement are arrays
    for array_key in ["strengths", "areas_for_improvement"]:
        if array_key not in feedback_json or not isinstance(feedback_json[array_key], list):
            feedback_json[array_key] = ["No specific points identified"]
    
    return feedback_json


def create_fallback_json(error_message):
    """
    Creates a fallback JSON structure with an error message.
    
    Returns:
        dict: Basic feedback JSON with error message
    """
    return {
        "overall_score": 0,
        "category_scores": {
            "Technical Knowledge": 0,
            "Communication": 0,
            "Problem Solving": 0,
            "Experience": 0
        },
        "strengths": ["Unable to determine"],
        "areas_for_improvement": ["Unable to determine"],
        "detailed_feedback": {
            "technical_skills": "Unable to evaluate",
            "communication_skills": "Unable to evaluate",
            "problem_solving": "Unable to evaluate",
            "experience_match": "Unable to evaluate"
        },
        "raw_feedback": error_message
    }


async def save_feedback_to_pdf(feedback, candidate_name):
    """
    Saves interview feedback to a PDF file.
    
    Args:
        feedback (str): JSON-formatted feedback string
        candidate_name (str): Name of the candidate
        
    Returns:
        str: Path to the generated PDF file
    """
    # Create a filename with timestamp
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"{candidate_name.replace(' ', '_')}_interview_feedback_{now}.pdf"
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create a custom style for headers
    styles.add(ParagraphStyle(
        name='Heading1', 
        fontName='Helvetica-Bold', 
        fontSize=14, 
        spaceAfter=12
    ))
    
    # Create a list to hold the PDF elements
    elements = []
    
    # Add title
    title = Paragraph(f"Interview Feedback for {candidate_name}", styles['Heading1'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    try:
        # Try to parse the feedback JSON
        if feedback is None:
            raise ValueError("Feedback is None")
            
        feedback_json = json.loads(feedback)
        # Add overall score
        if "overall_score" in feedback_json:
            overall = Paragraph(f"Overall Score: {feedback_json['overall_score']}/10", styles['Normal'])
            elements.append(overall)
            elements.append(Spacer(1, 12))
        
        # Add category scores if available
        if "category_scores" in feedback_json and feedback_json["category_scores"] is not None:
            add_category_scores_table(elements, feedback_json, styles)
        
        # Add detailed feedback
        if "detailed_feedback" in feedback_json and feedback_json["detailed_feedback"] is not None:
            add_detailed_feedback(elements, feedback_json, styles)
        
        # Add strengths and areas for improvement
        add_strengths_and_improvements(elements, feedback_json, styles)
                
        # Add raw feedback as a summary
        if "raw_feedback" in feedback_json and feedback_json["raw_feedback"] is not None:
            elements.append(Paragraph("Overall Assessment:", styles['Heading1']))
            elements.append(Spacer(1, 6))
            elements.append(Paragraph(feedback_json["raw_feedback"], styles['Normal']))
            
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        # Handle cases where feedback isn't valid JSON or is missing expected keys
        print(f"Error processing feedback JSON: {e}", file=sys.stderr)
        add_raw_feedback(elements, feedback, styles)
    
    # Build the PDF
    doc.build(elements)
    print(f"Feedback saved to {pdf_filename}")
    return pdf_filename


def add_category_scores_table(elements, feedback_json, styles):
    """Adds a table of category scores to the PDF elements."""
    categories = list(feedback_json["category_scores"].keys())
    scores = [feedback_json["category_scores"][cat] for cat in categories]
    
    # Create a table for category scores
    data = [["Category", "Score"]]
    for i in range(len(categories)):
        data.append([categories[i], scores[i]])
    
    table = Table(data, colWidths=[400, 100])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
        ('BACKGROUND', (0, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(Paragraph("Category Scores:", styles['Heading1']))
    elements.append(Spacer(1, 6))
    elements.append(table)
    elements.append(Spacer(1, 12))


def add_detailed_feedback(elements, feedback_json, styles):
    """Adds detailed feedback sections to the PDF elements."""
    elements.append(Paragraph("Detailed Feedback:", styles['Heading1']))
    elements.append(Spacer(1, 6))
    
    for section, content in feedback_json["detailed_feedback"].items():
        # Convert snake_case to Title Case for display
        section_title = ' '.join(word.capitalize() for word in section.split('_'))
        elements.append(Paragraph(section_title, styles['Heading2']))
        elements.append(Paragraph(content, styles['Normal']))
        elements.append(Spacer(1, 10))


def add_strengths_and_improvements(elements, feedback_json, styles):
    """Adds strengths and areas for improvement to the PDF elements."""
    for section in ["strengths", "areas_for_improvement"]:
        if section in feedback_json and feedback_json[section] is not None:
            title = "Strengths" if section == "strengths" else "Areas for Improvement"
            elements.append(Paragraph(title, styles['Heading1']))
            elements.append(Spacer(1, 6))
            
            for item in feedback_json[section]:
                elements.append(Paragraph(f"• {item}", styles['Normal']))
            
            elements.append(Spacer(1, 12))


def add_raw_feedback(elements, feedback, styles):
    """Adds raw feedback text when structured data is unavailable."""
    elements.append(Paragraph("Interview Feedback:", styles['Heading1']))
    elements.append(Spacer(1, 6))
    
    # Use the raw feedback string
    if feedback:
        elements.append(Paragraph(str(feedback), styles['Normal']))
    else:
        elements.append(Paragraph(
            "No feedback data was generated for this interview. "
            "This may be due to an error in processing the interview responses.", 
            styles['Normal']
        ))
