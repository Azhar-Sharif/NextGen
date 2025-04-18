import asyncio
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
import configparser
import os
import random
import re
import sys

# Load API keys from config.ini
current_dir = os.path.dirname(__file__)
config_path = os.path.join(current_dir, "config.ini")
config = configparser.ConfigParser()
config.read(config_path)
openai_api_key = config["openai"]["api_key"]

def initialize_llm(api_key, model_name="gpt-3.5-turbo"):
    """Initializes the GPT-4 model using the provided API key and model name."""
    return ChatOpenAI(temperature=0.7, openai_api_key=api_key, model=model_name)

# Define prompt templates for each question type
question_templates = {
    "ice-breaking": """
        You are an AI interview assistant for a data science role. Your task is to generate a warm and engaging ice-breaking question that connects to the candidate's experience in data science.
        The candidate is applying for the following role: {job_description}.
        The candidate's experience is as follows: {user_experience}.

        Based on the conversation history:

        {chat_history}

        ask a relevant ice-breaking question.

        The question must:
        - Start with a friendly opening like "Hi [candidate's name]," or "It's great to meet you, [candidate's name]."
        - Relate to their data science journey, such as a project that sparked their interest or a challenge they overcame.
        - Be open-ended to encourage them to share more about their passion for data science.
        - Be concise and easy to understand.
        - Be non-repetitive, different from any questions already asked.
        - Focus on their data science interests or a specific part of their background that stands out.

        Already asked questions: {asked_questions}

        ONLY OUTPUT THE QUESTION NOTHING ELSE
    """,
    "technical": """
        You are an AI interview assistant for a data science role. Your task is to generate a technical question that assesses the candidate's understanding of key data science concepts.
        The candidate is applying for the following role: {job_description}.
        The candidate's experience is as follows: {user_experience}.

        Based on the conversation history:

        {chat_history}

        ask a relevant technical question.

        Focus on these data science areas:
        - Statistical methods and hypothesis testing
        - Machine learning algorithms and model selection
        - Feature engineering and selection techniques
        - Data preprocessing and cleaning
        - Model evaluation metrics and validation strategies
        - Python libraries (NumPy, Pandas, Scikit-learn, TensorFlow/PyTorch)
        - SQL and database knowledge
        - Big data technologies (Spark, Hadoop)
        - Data visualization (Matplotlib, Seaborn, Plotly)
        - Experiment design and A/B testing
        - Time series analysis
        - Natural Language Processing
        - Deep Learning architectures
        
        The question must:
        - Match the candidate's experience level (basic for juniors, advanced for seniors)
        - Be scenario-based when possible (e.g., "How would you handle class imbalance in a fraud detection model?")
        - Encourage explanation of methodology or approach
        - Be concise and clearly phrased
        - Be non-repetitive, different from any questions already asked
        - Progressively increase in difficulty based on previous answers
        - Adjust based on the candidate's previous responses - if they answered well, ask deeper questions
        
        Already asked questions: {asked_questions}

        ONLY OUTPUT THE QUESTION NOTHING ELSE
    """,
    "behavioral": """
        You are an AI interview assistant for a data science role. Your task is to generate a behavioral question that assesses the candidate's past experiences in data science contexts.
        The candidate is applying for the following role: {job_description}.
        The candidate's experience is as follows: {user_experience}.

        Based on the conversation history:

        {chat_history}

        ask a relevant behavioral question.

        The question must:
        - Focus on data science-specific scenarios like:
          * Collaborating with cross-functional teams
          * Explaining technical concepts to non-technical stakeholders
          * Dealing with ambiguous requirements
          * Managing large-scale data projects
          * Handling tight deadlines
          * Learning new technologies rapidly
          * Working with incomplete or messy data
          * Resolving conflicts in analytical approaches
        - Use prompts like "Tell me about a time when..." or "Describe a situation where..."
        - Be relevant to the job description and level
        - Be open-ended to encourage specific examples using the STAR method
        - Be concise and easy to understand
        - Be non-repetitive, different from any questions already asked

        Already asked questions: {asked_questions}

        ONLY OUTPUT THE QUESTION NOTHING ELSE
    """,
    "problem-solving": """
        You are an AI interview assistant for a data science role. Your task is to generate a problem-solving question that assesses the candidate's analytical and critical thinking skills in a data science context.
        The candidate is applying for the following role: {job_description}.
        The candidate's experience is as follows: {user_experience}.

        Based on the conversation history:

        {chat_history}

        ask a relevant problem-solving question.

        The question must:
        - Present a realistic data science scenario such as:
          * Dealing with missing data in a critical dataset
          * Selecting appropriate models for specific business problems
          * Setting up an experimentation framework
          * Handling overfitting/underfitting
          * Interpreting confusing model results
          * Designing a data pipeline for a specific use case
          * Optimizing model performance
          * Building an ML system with specific constraints
        - Be relevant to the job description and adapted to the candidate's experience level
        - Encourage step-by-step approach and trade-off considerations
        - Be concrete enough to allow for specific answers but open enough for creativity
        - Be concise and clearly defined
        - Be non-repetitive, different from any questions already asked

        Already asked questions: {asked_questions}

        ONLY OUTPUT THE QUESTION NOTHING ELSE
    """,
    "career-goal": """
        You are an AI interview assistant for a data science role. Your task is to generate a question that explores the candidate's career goals and motivation within the data science field.
        The candidate is applying for the following role: {job_description}.
        The candidate's experience is as follows: {user_experience}.

        Based on the conversation history:

        {chat_history}

        ask a relevant question about their career goals or motivation.

        The question must:
        - Focus on their specific data science aspirations such as:
          * Areas of specialization (ML engineering, research, analytics)
          * Technologies they're interested in mastering
          * How they stay current with data science developments
          * Long-term vision in the rapidly evolving field
          * What impact they hope to make with data science
          * Alignment between their goals and the company's mission
        - Relate to the specific job description
        - Be open-ended to encourage reflection on how the role fits into their career path
        - Be concise and easy to understand
        - Be non-repetitive, different from any questions already asked

        Already asked questions: {asked_questions}

        ONLY OUTPUT THE QUESTION NOTHING ELSE
    """,
}

# Initialize LLM with GPT-4
llm = initialize_llm(openai_api_key, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

async def generate_question(
    job_description,
    user_experience,
    asked_questions,
    chat_history,
    question_type,
    covered_topics=[],
    difficulty_level="intermediate"
):
    """Generates a contextually relevant interview question based on the question type and difficulty level asynchronously."""
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # Get the last user response and feedback for context
            last_response = None
            last_feedback = None
            previous_questions = []
            
            # Extract previous questions and responses for context
            for i, message in enumerate(chat_history):
                if message.type == "ai" and not message.content.startswith("Feedback:"):
                    previous_questions.append(message.content)
                if message.type == "human" and i > 0:
                    last_response = message.content
                elif message.type == "ai" and message.content.startswith("Feedback:"):
                    last_feedback = message.content[9:].strip()

            # Limit previous questions to last 3 for context
            previous_questions = previous_questions[-3:] if previous_questions else []
            
            if question_type not in question_templates:
                print(f"Invalid question type: {question_type}")
                return "Unable to generate a question."

            # Extract topics from previous responses to avoid repetition
            response_topics = extract_topics_from_responses(chat_history)
            
            avoid_topics_instruction = ""
            if covered_topics or response_topics:
                all_covered_topics = list(set(covered_topics + response_topics))
                avoid_topics_instruction = f"The candidate has already covered the following topics: {', '.join(all_covered_topics)}. Avoid these specific topics."

            context_aware_instruction = ""
            if last_response and last_feedback:
                # Determine if we should follow up on the previous topic or change topics
                if "lacks depth" in last_feedback.lower() or "should provide" in last_feedback.lower() or "could elaborate" in last_feedback.lower():
                    # The candidate didn't answer well, so we might want to follow up or simplify
                    context_aware_instruction = f"""
                    Previous question: {previous_questions[-1] if previous_questions else ""}
                    Previous response: {last_response}
                    Response feedback: {last_feedback}

                    Based on this context:
                    - The candidate struggled with the previous question
                    - Either ask a simplified version of the same concept OR
                    - Move to a different topic that might be easier to grasp
                    - Do NOT repeat the exact same question
                    """
                elif any(pos_term in last_feedback.lower() for pos_term in ["excellent", "good", "strong", "clear", "solid"]):
                    # The candidate answered well, so we can go deeper or increase difficulty
                    context_aware_instruction = f"""
                    Previous question: {previous_questions[-1] if previous_questions else ""}
                    Previous response: {last_response}
                    Response feedback: {last_feedback}

                    Based on this context:
                    - The candidate answered well
                    - Build on their knowledge by asking a more challenging follow-up question
                    - Explore a related but more advanced concept
                    - Do NOT repeat the exact same question
                    """
                else:
                    # Neutral feedback, so move to a new topic
                    context_aware_instruction = f"""
                    Previous question: {previous_questions[-1] if previous_questions else ""}
                    Previous response: {last_response}
                    Response feedback: {last_feedback}

                    Based on this context:
                    - Move to a new data science topic
                    - Ensure the question is at an appropriate difficulty level
                    - Do NOT repeat the exact same question or concept
                    """
                
            # Add difficulty level instructions
            difficulty_instruction = ""
            if question_type == "technical":
                if difficulty_level == "basic":
                    difficulty_instruction = """
                    Generate a BASIC level technical question suitable for entry-level data scientists:
                    - Focus on fundamental concepts (e.g., basic statistics, simple ML models, data cleaning)
                    - Use clear, straightforward language without complex jargon
                    - Focus on definition, understanding, and basic application
                    - Example topics: data preprocessing, basic metrics, types of ML algorithms, simple statistics
                    """
                elif difficulty_level == "intermediate":
                    difficulty_instruction = """
                    Generate an INTERMEDIATE level technical question suitable for data scientists with 2-5 years experience:
                    - Focus on application of concepts rather than just definitions
                    - Include scenario-based questions requiring analysis
                    - Cover more specialized topics like feature engineering, model selection, or validation strategies
                    - Example topics: handling imbalanced data, cross-validation approaches, feature importance, model interpretability
                    """
                elif difficulty_level == "advanced":
                    difficulty_instruction = """
                    Generate an ADVANCED level technical question suitable for senior data scientists:
                    - Focus on complex, nuanced scenarios requiring deep expertise
                    - Include questions about trade-offs, architectural decisions, and best practices
                    - Cover advanced topics like model deployment, MLOps, ethics, or specialized algorithms
                    - Example topics: model drift handling, MLOps pipelines, complex ensemble techniques, deep learning architectures
                    """

            # Add strict instruction to prevent repetition
            avoid_repetition = """
            IMPORTANT: Your question MUST be significantly different from these previous questions:
            {}
            
            DO NOT repeat the same concepts, phrasing, or examples. Generate a fresh, original question.
            """.format("\n".join(f"- {q}" for q in previous_questions))
            
            # Add explicit format instruction to ensure clean output
            format_instruction = """
            EXTREMELY IMPORTANT OUTPUT FORMAT INSTRUCTIONS:
            1. Output ONLY the interview question itself, nothing else
            2. Do NOT include any bullets, numbering, or formatting
            3. Do NOT include any instructions, explanations or metadata
            4. Do NOT prefix the question with phrases like "Question:"
            5. The output should be a single, clean, professional interview question
            6. Do NOT include example answers or follow-up instructions
            
            GOOD EXAMPLE OUTPUT: "Can you explain the difference between supervised and unsupervised learning?"
            BAD EXAMPLE OUTPUT: "- Here is a basic question about machine learning: Can you explain the difference between supervised and unsupervised learning?"
            """

            question_template = f"""
            {question_templates[question_type]}

            {difficulty_instruction}

            {context_aware_instruction}
            
            {avoid_repetition}
            
            {avoid_topics_instruction}
            
            {format_instruction}
            """.format(
                job_description=job_description,
                user_experience=user_experience,
                chat_history="{chat_history}",
                asked_questions=asked_questions,
                avoid_topics_instruction=avoid_topics_instruction,
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", question_template),
                    MessagesPlaceholder(variable_name="chat_history"),
                ]
            )

            chain = (
                {
                    "job_description": RunnablePassthrough(),
                    "user_experience": RunnablePassthrough(),
                    "asked_questions": RunnablePassthrough(),
                    "question_type": RunnablePassthrough(),
                    "avoid_topics_instruction": RunnablePassthrough(),
                    "difficulty_level": RunnablePassthrough(),
                    "chat_history": lambda x: x["chat_history"],
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            input_data = {
                "job_description": job_description,
                "user_experience": user_experience,
                "asked_questions": asked_questions,
                "chat_history": chat_history,
                "question_type": question_type,
                "avoid_topics_instruction": avoid_topics_instruction,
                "difficulty_level": difficulty_level,
            }

            response = await chain.ainvoke(input_data)
            # Clean the question of any potential formatting/prefixes
            question = clean_generated_question(response.strip())

            if not question:
                print("Empty question generated.")
                raise ValueError("Empty question generated")
                
            # Verify this isn't a duplicate question
            if is_similar_to_previous(question, asked_questions):
                print("Generated a similar question to one already asked. Retrying...")
                raise ValueError("Similar question already asked")

            return question

        except Exception as e:
            print(
                f"Error generating question (Attempt {attempt + 1}/{max_retries}): {e}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (1 + random.uniform(0, 1)))
                retry_delay *= 2
            else:
                print("Max retries reached. Unable to generate a question.")
                return "Unable to generate a question."

    return "Unable to generate a question."

def clean_generated_question(question):
    """Clean the generated question to ensure it's in the correct format."""
    # Remove any bullet points
    if question.startswith('- '):
        question = question[2:]
    if question.startswith('* '):
        question = question[2:]
        
    # Remove any numbering
    if re.match(r'^\d+\.', question):
        question = re.sub(r'^\d+\.\s*', '', question)
        
    # Remove any "Question:" prefix
    patterns_to_remove = [
        r'^Question:\s*',
        r'^Technical Question:\s*',
        r'^Here\'s a question:\s*',
        r'^Interview Question:\s*',
        r'^Data Science Question:\s*',
    ]
    
    for pattern in patterns_to_remove:
        question = re.sub(pattern, '', question, flags=re.IGNORECASE)
    
    # Remove any instructions
    lines = question.split('\n')
    clean_lines = []
    for line in lines:
        if line.strip() and not line.strip().startswith('-') and not line.strip().startswith('*'):
            clean_lines.append(line.strip())
    
    # Rejoin and return
    return ' '.join(clean_lines)

def extract_topics_from_responses(chat_history):
    """Extract main topics from previous responses to avoid repeating them."""
    topics = []
    for message in chat_history:
        if message.type == "human":
            # Look for key phrases that might indicate topics
            content = message.content.lower()
            for keyword in ["about", "regarding", "on", "using", "with"]:
                if keyword in content:
                    parts = content.split(keyword)
                    if len(parts) > 1:
                        # Extract phrases after the keyword
                        potential_topic = parts[1].split(".")[0].split(",")[0].strip()
                        if 3 < len(potential_topic) < 30:  # Reasonable length for a topic
                            topics.append(potential_topic)
    
    # Return unique topics
    return list(set(topics))

def is_similar_to_previous(new_question, previous_questions, similarity_threshold=0.7):
    """Check if a new question is too similar to previously asked questions."""
    for prev_q in previous_questions:
        # Convert to lowercase for comparison
        new_lower = new_question.lower()
        prev_lower = prev_q.lower()
        
        # Simple word overlap check
        new_words = set(new_lower.split())
        prev_words = set(prev_lower.split())
        
        if len(new_words) == 0 or len(prev_words) == 0:
            continue
            
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(new_words.intersection(prev_words))
        union = len(new_words.union(prev_words))
        similarity = intersection / union
        
        if similarity > similarity_threshold:
            return True
            
    return False

async def should_ask_more_questions(
    conversation_history, asked_questions, job_description, user_experience, question_type
):
    """Determines if more questions should be asked based on existing criteria asynchronously."""
    prompt = f"""
    You are an AI interview assistant for a Data Science position. You are conducting an interview for: {job_description}.
    The candidate's experience is as follows: {user_experience}
    
    Here is the conversation history so far:

    "{conversation_history}"

    Technical questions already asked: {[q for q in asked_questions if q.strip()]}

    You are currently in the {question_type} section of the interview.

    Your role is to thoroughly evaluate the candidate's technical skills for a DATA SCIENCE position.

    Evaluate the candidate's responses against these data science-specific criteria:

    1. TECHNICAL DEPTH:
       - Has the candidate demonstrated understanding of statistical methods, machine learning algorithms, feature engineering, and model evaluation?
       - Did they explain their approach with appropriate technical terminology?
       - Did they discuss tradeoffs between different methodologies?

    2. PRACTICAL APPLICATION:
       - Has the candidate shown how they apply data science concepts to solve real-world problems?
       - Did they mention specific tools/libraries (Python, pandas, scikit-learn, TensorFlow/PyTorch, SQL, etc.)?
       - Did they describe their workflow and methodologies?

    3. ANALYTICAL THINKING:
       - Did the candidate demonstrate critical thinking when analyzing problems?
       - Did they consider data quality, biases, and limitations in their approach?
       - Did they show systems thinking about how models fit into larger business contexts?

    4. KNOWLEDGE GAPS:
       - Are there key areas of data science not yet explored (ML algorithms, NLP, deep learning, time series, etc.)?
       - Are there technical skills mentioned in the job description that haven't been discussed?
       - Has the candidate shown any knowledge gaps that require further exploration?

    5. RESPONSE QUALITY:
       - Were the answers clear, structured, and showed depth of understanding?
       - Did the candidate handle increasingly difficult questions well?
       - Is there a need to probe deeper into any specific area?

    DECISION LOGIC:
    - If the candidate has NOT been thoroughly evaluated on ALL data science fundamentals relevant to the job description, respond with 'yes'
    - If the candidate has shown GAPS in knowledge that need further exploration, respond with 'yes'
    - If you have NOT asked questions covering different difficulty levels to assess depth, respond with 'yes'
    - If ALL key areas from the job description have been covered AND responses show adequate depth, respond with 'no'
    
    Think step by step about each criteria before making your decision.
    Respond with ONLY 'yes' if more questions are needed, or 'no' if enough information has been gathered.
    """
    try:
        response = await llm.ainvoke(prompt)
        decision = response.content.strip().lower()
        
        # Log decision to stderr instead of stdout to avoid showing in the interview
        print(f"LLM decision to ask more questions: {decision}", file=sys.stderr)
        
        # Extract just the yes/no decision
        if 'yes' in decision.lower():
            return True
        elif 'no' in decision.lower():
            return False
        else:
            # Default to yes if unclear
            return True
    except Exception as e:
        print(f"Error determining whether to ask more questions: {e}", file=sys.stderr)
        return True  # Default to asking more questions if there's an error