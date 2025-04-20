import asyncio
from langchain_openai import ChatOpenAI
import aiofiles
import configparser
import os
import tempfile
import pygame.mixer
import boto3
from dotenv import load_dotenv
from .audio_processing import record_audio_async, save_audio_to_wav_async, transcribe_audio_with_whisper_async
from .question_generation import generate_question, should_ask_more_questions, initialize_llm, memory
from .feedback_generation import generate_overall_feedback, save_feedback_to_pdf
from .utils import extract_name_from_text
from .response_analysis import ResponseAnalyzer, add_analysis_to_history
from .rag_engine import RAGEngine
from openai import AsyncOpenAI
import sys
import json
import shutil

class Interviewer:
    def __init__(self, config_file=None, debug=False):
        """
        Initialize the Interviewer.

        Args:
            config_file (str, optional): Path to a custom config file
            debug (bool): Whether to enable debug logging
        """
        self.debug = debug

        # Load configuration
        config = configparser.ConfigParser()
        if config_file and os.path.exists(config_file):
            config.read(config_file)
        else:
            config.read('backend/interview_project/config.ini')

        # Load environment variables from .env file as a fallback
        load_dotenv()

        # Configure AWS services
        self.aws_region_name = config.get('aws', 'region_name', fallback=os.getenv('AWS_REGION_NAME', 'us-east-1'))
        self.polly_voice_id = config.get('aws', 'polly_voice_id', fallback=os.getenv('AWS_POLLY_VOICE_ID', 'Joanna'))
        self.openai_api_key = config.get('openai', 'api_key', fallback=os.getenv('OPENAI_API_KEY'))

        # Initialize AWS Polly client
        self.polly_client = self.initialize_polly_client()

        # Initialize components
        self.response_analyzer = ResponseAnalyzer()
        self.openai_client = self.initialize_llm(self.openai_api_key)

        # Interview state
        self.user_experience = ""
        self.user_name = "User"
        self.asked_questions = []
        self.current_difficulty = "basic"
        self.answers = []
        self.job_description = ""
        self.rag_engine = None  # Will be initialized later

        # Clear the audio directory for the new interview instance
        self.clear_audio_directory()

        if self.debug:
            print("Interviewer initialized with debug mode enabled.", file=sys.stderr)
        print("Interviewer initialized.")

    def clear_audio_directory(self):
        """Clears the static/audio/ directory to reset question audio for a new interview instance."""
        audio_dir = os.path.join(os.getcwd(), "frontend", "static", "audio")
        if os.path.exists(audio_dir):
            try:
                shutil.rmtree(audio_dir)  # Remove the directory and its contents
                os.makedirs(audio_dir)  # Recreate the directory
                print("Audio directory cleared and reset for the new interview instance.", file=sys.stderr)
            except Exception as e:
                print(f"Error clearing audio directory: {e}", file=sys.stderr)

    def initialize_polly_client(self):
        """Initializes the Amazon Polly client."""
        try:
            return boto3.Session(region_name=self.aws_region_name).client("polly")
        except Exception as e:
            print(f"Error initializing Polly client: {e}", file=sys.stderr)
            return None

    @staticmethod
    def initialize_llm(api_key, model_name="gpt-3.5-turbo"):
        """Initializes the GPT model using the provided API key and model name."""
        return ChatOpenAI(temperature=0.7, openai_api_key=api_key, model=model_name)

    async def speak_text_with_polly(self, text):
        """Converts text to speech using Amazon Polly and plays it with pygame."""
        loop = asyncio.get_event_loop()
        temp_audio_path = None
        try:
            # Synthesize speech with Polly (run in executor)
            response = await loop.run_in_executor(
                None,
                lambda: self.polly_client.synthesize_speech(
                    VoiceId=self.polly_voice_id,
                    OutputFormat="mp3",
                    Text=text
                )
            )
            # Write audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_path = temp_audio_file.name
                temp_audio_file.write(response["AudioStream"].read())

            # Load and play audio with pygame (run in executor for loading)
            await loop.run_in_executor(None, pygame.mixer.music.load, temp_audio_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish asynchronously
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)  # Non-blocking wait
        except Exception as e:
            print(f"Error with Polly speech synthesis: {e}")
        finally:
            # Cleanup temp file
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception as e:
                    print(f"Error cleaning up temp file: {e}")

    async def get_job_description(self,job_title):
        
        self.job_description = job_title  # Synchronous; consider aioconsole for async input

    async def get_user_experience(self,experience_text):
            self.user_experience = await self.determine_experience_level(experience_text)
    
    async def initialize_rag_engine(self):
        """Initialize the RAG engine for technical question generation and answer evaluation."""
        try:
            print("Initializing RAG engine...", file=sys.stderr)
            # Paths to required files
            current_dir = os.path.dirname(__file__)
            index_path = os.path.join(current_dir, "faiss_index.bin")
            chunks_path = os.path.join(current_dir, "cleaned_chunks.json")
            
            # Create the RAG engine
            self.rag_engine = await RAGEngine.create_engine(
                index_path=index_path,
                chunks_path=chunks_path
            )
            print("RAG engine initialized successfully", file=sys.stderr)
            return True
        except Exception as e:
            print(f"Error initializing RAG engine: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return False

    async def conduct_interview(self):
        async for question_data in self.conduct_fixed_questioning("ice-breaking", num_questions=4):
            print("question_number", question_data["loop value"])
            yield question_data  # Yield the question data (text and audio)

            # Record and process the user's answer
            audio_file_path = "temp/blob"  # Replace with the actual path
            user_response, feedback = await self.record_and_process_answer(audio_file_path)
            if user_response:
                print(f"User's response: {user_response}")
                print(f"Feedback: {feedback}")
            else:
                print(f"Error or no response: {feedback}")

    async def conduct_fixed_questioning(self, question_type, num_questions):
        """Asks a fixed number of questions of a specific type asynchronously."""
        covered_topics = []

        for i in range(num_questions):
            print(f"Asking question {i + 1} of {num_questions}", file=sys.stderr)  # Debugging
            history = memory.chat_memory.messages
            question = await generate_question(
                self.job_description,
                self.user_experience,
                self.asked_questions,
                history,
                question_type,
                covered_topics=covered_topics,
            )
            if question == "Unable to generate a question.":
                print(f"Unable to generate a {question_type} question.", file=sys.stderr)
                break

            # Ask the question and generate audio
            question_text, question_audio = await self.ask_question(question, question_type)
            if question_text and question_audio:
                if i==2:
                    break
                yield {"text": question_text, "audio": question_audio, "loop value": i}
            else:
                print(f"Failed to ask {question_type} question {i + 1}.", file=sys.stderr)
                break

    async def ask_question(self, question, question_type):
        """Asks a question and generates the audio asynchronously, returning the question text and audio."""
        try:
            # Clean the question from any internal instructions that might be included
            cleaned_question = self.clean_question_for_speech(question)

            if self.user_name != "User":
                first_question_mark_index = cleaned_question.find("?")
                if first_question_mark_index != -1:
                    personalized_question = (
                        cleaned_question[:first_question_mark_index]
                        + f", {self.user_name}"
                        + cleaned_question[first_question_mark_index:]
                    )
                else:
                    personalized_question = cleaned_question + f", {self.user_name}"
            else:
                personalized_question = cleaned_question

            # Print only once before speaking
            print(f"AI: {personalized_question}")

            # Generate audio for the question
            loop = asyncio.get_event_loop()
            temp_audio_path = None
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.polly_client.synthesize_speech(
                        VoiceId=self.polly_voice_id,
                        OutputFormat="mp3",
                        Text=personalized_question
                    )
                )
                # Save audio to the static/audio/ directory
                audio_dir = os.path.join(os.getcwd(), "frontend", "static", "audio")
                os.makedirs(audio_dir, exist_ok=True)
                question_number = len(self.asked_questions) + 1
                audio_filename = os.path.join(audio_dir, f"question_{question_number}_{question_type}.mp3")
                with open(audio_filename, "wb") as audio_file:
                    audio_file.write(response["AudioStream"].read())
            except Exception as e:
                print(f"Error generating audio for question: {e}", file=sys.stderr)
                audio_filename = None

            # Add the question to the asked questions list
            if question not in self.asked_questions:
                self.asked_questions.append(question)

            return personalized_question, f"/static/audio/{os.path.basename(audio_filename)}"
        except Exception as e:
            print(f"Error asking question: {e}", file=sys.stderr)
            return None, None

    async def conduct_adaptive_questioning(self, question_type, min_questions=0, max_questions=10):
        """Conducts questioning with adaptive difficulty levels based on candidate responses."""
        question_count = 0
        consecutive_good_responses = 0
        consecutive_poor_responses = 0
        
        # Track topics already covered to avoid repetition
        covered_topics = []
        
        # For internal logging only, not shown to user
        print(f"Starting {question_type} questions at {self.current_difficulty} difficulty level", file=sys.stderr)
        
        # Initialize RAG engine if it's a technical question and not already initialized
        if question_type == "technical" and not self.rag_engine:
            rag_initialized = await self.initialize_rag_engine()
            if not rag_initialized:
                print("Failed to initialize RAG engine. Falling back to standard question generation.", file=sys.stderr)
        
        while question_count < max_questions:
            try:
                history = memory.chat_memory.messages
                
                # Use RAG for technical questions if available
                if (question_type == "technical" and self.rag_engine):
                    # Get previously asked technical questions
                    previous_technical_questions = [q for q in self.asked_questions 
                                                  if q in [msg.content for msg in history if hasattr(msg, 'type') and msg.type == "ai"]]
                    
                    # Generate question using RAG
                    question = await self.rag_engine.generate_technical_question(
                        job_description=self.job_description,
                        user_experience=self.user_experience,
                        difficulty=self.current_difficulty,
                        previous_questions=previous_technical_questions,
                        openai_client=self.openai_client
                    )
                    
                    print(f"Generated RAG-based technical question: {question}", file=sys.stderr)
                else:
                    # Add difficulty level to the context for question generation
                    question = await generate_question(
                        self.job_description,
                        self.user_experience,
                        self.asked_questions,
                        history,
                        question_type,
                        covered_topics=covered_topics,
                        difficulty_level=self.current_difficulty
                    )
                
                if question == "Unable to generate a question.":
                    print(f"Unable to generate further {question_type} questions.", file=sys.stderr)
                    break
                        
                # Ask the question and record if it was successful
                if await self.ask_question(question):
                    question_count += 1
                    
                    # Get the most recent feedback to adjust difficulty
                    last_feedback = self.get_last_feedback()
                    # For internal logging only
                    print(f"Feedback analysis: {last_feedback}", file=sys.stderr)
                    
                    # Extract topics from the response to track covered material
                    latest_response = self.get_last_response()
                    if latest_response:
                        extracted_topics = self.extract_response_topics(latest_response)
                        covered_topics.extend(extracted_topics)
                        # For internal logging only
                        print(f"Added topics: {extracted_topics}", file=sys.stderr)
                    
                    # Adjust difficulty based on response quality
                    if self.is_positive_feedback(last_feedback):
                        consecutive_good_responses += 1
                        consecutive_poor_responses = 0
                        
                        # For internal logging only
                        print(f"Positive feedback. Consecutive good: {consecutive_good_responses}", file=sys.stderr)
                        
                        # Increase difficulty after 2 consecutive good responses
                        if consecutive_good_responses >= 2:
                            old_difficulty = self.current_difficulty
                            if self.current_difficulty == "basic":
                                self.current_difficulty = "intermediate"
                            elif self.current_difficulty == "intermediate":
                                self.current_difficulty = "advanced"
                                    
                            if old_difficulty != self.current_difficulty:
                                # For internal logging only
                                print(f"Increasing difficulty to {self.current_difficulty}", file=sys.stderr)
                            consecutive_good_responses = 0
                    else:
                        consecutive_poor_responses += 1
                        consecutive_good_responses = 0
                        
                        # For internal logging only
                        print(f"Needs improvement. Consecutive poor: {consecutive_poor_responses}", file=sys.stderr)
                        
                        # Decrease difficulty after 2 consecutive poor responses
                        if consecutive_poor_responses >= 2:
                            old_difficulty = self.current_difficulty
                            if self.current_difficulty == "advanced":
                                self.current_difficulty = "intermediate"
                            elif self.current_difficulty == "intermediate":
                                self.current_difficulty = "basic"
                                    
                            if old_difficulty != self.current_difficulty:
                                # For internal logging only
                                print(f"Decreasing difficulty to {self.current_difficulty}", file=sys.stderr)
                            consecutive_poor_responses = 0
                else:
                    print("Failed to record answer. Moving to next question.", file=sys.stderr)
                    break
                        
                # Check if we should continue asking questions
                if question_count >= min_questions:
                    history = memory.chat_memory.messages
                    should_continue = await should_ask_more_questions(
                        history,
                        self.asked_questions,
                        self.job_description,
                        self.user_experience,
                        question_type
                    )
                    if not should_continue:
                        # For internal logging only
                        print(f"Moving on from {question_type} questions after {question_count} questions.", file=sys.stderr)
                        break
            except Exception as e:
                print(f"Error in questioning: {e}", file=sys.stderr)
                break

    
    def clean_question_for_speech(self, question):
        """Cleans the question to remove any model instructions or non-user-facing content."""
        # Remove common instruction patterns
        if "Generate a" in question and "level technical question" in question:
            # Handle cases where the entire prompt is returned
            examples = question.split('Example')
            if len(examples) > 1:
                for part in examples[1:]:
                    colon_split = part.split(':')
                    if len(colon_split) > 1:
                        cleaned = colon_split[1].strip()
                        # Extract content from quotes if present
                        if '"' in cleaned:
                            quoted = cleaned.split('"')
                            if len(quoted) > 1:
                                return quoted[1].strip()
                        else:
                            return cleaned
        
        # Remove any bullet points and instruction-like content
        lines = question.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip instruction-like lines
            if (line.startswith('-') or line.startswith('*') or 
                "Focus on" in line or "example" in line.lower() or
                "generate" in line.lower() or "basic" in line.lower() and "level" in line.lower()):
                continue
            if line:
                cleaned_lines.append(line)
        
        # Join the remaining content
        cleaned_question = ' '.join(cleaned_lines)
        
        # Remove any "Question:" prefix
        prefixes = ["question:", "technical question:", "here's a question:", 
                    "interview question:", "data science question:"]
        for prefix in prefixes:
            if cleaned_question.lower().startswith(prefix):
                cleaned_question = cleaned_question[len(prefix):].strip()
        
        # If we filtered too much, return the original content
        if len(cleaned_question) < 10 and len(question) > 10:
            # Final attempt to extract a question - look for a question mark
            if "?" in question:
                # Find sentences with question marks
                sentences = question.split(".")
                for sentence in sentences:
                    if "?" in sentence and len(sentence.strip()) > 10:
                        return sentence.strip()
            return question
        
        return cleaned_question

    async def generate_and_save_feedback(self):
        """Generates overall feedback and saves it to a PDF asynchronously."""
        try:
            if not self.asked_questions or not self.answers:
                print("No questions or answers to generate feedback from.", file=sys.stderr)
                error_message = "Unable to generate feedback: No valid interview data was collected."
                print(f"AI: {error_message}")
                return None
            
            overall_feedback = await generate_overall_feedback(
                self.openai_api_key,
                self.asked_questions,
                self.answers,
                self.job_description,
                self.user_experience,
                self.rag_engine,
            )
            
            if not overall_feedback:
                print("No feedback was generated.", file=sys.stderr)
                error_message = "I'm sorry, but I couldn't generate feedback for this interview session."
                print(f"AI: {error_message}")
                return None
            
            try:
                # Check if it's valid JSON
                interview_result_in_json = json.loads(overall_feedback)
                return interview_result_in_json
            except json.JSONDecodeError:
                print("Generated feedback is not valid JSON.", file=sys.stderr)
                # Continue anyway, as our save_feedback_to_pdf can handle non-JSON strings
            """
            # Generate the PDF with the feedback
            pdf_file = await save_feedback_to_pdf(overall_feedback, self.user_name)
            
            if pdf_file:
                success_message = f"Feedback saved to PDF: {pdf_file}"
                print(success_message)
                return pdf_file
            else:
                error_message = "Unable to save feedback to PDF."
                print(error_message, file=sys.stderr)
                return None
            """
        except Exception as e:
            error_message = f"Error generating or saving feedback: {str(e)}"
            print(error_message, file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return None
        
    def get_last_feedback(self):
        """Extracts the most recent feedback from the conversation history."""
        for message in reversed(memory.chat_memory.messages):
            if hasattr(message, 'content') and message.content.startswith("Feedback:"):
                return message.content[9:].trip()
        return ""

    def get_last_response(self):
        """Gets the most recent user response from the conversation history."""
        for message in reversed(memory.chat_memory.messages):
            if hasattr(message, 'type') and message.type == "human":
                return message.content
        return ""

    def extract_response_topics(self, response):
        """Extract key topics from a user response to track what's been covered."""
        topics = []
        
        # Simple keyword extraction
        key_data_science_terms = [
            "regression", "classification", "clustering", "neural network", 
            "machine learning", "deep learning", "data cleaning", "feature engineering",
            "statistical", "hypothesis", "model selection", "cross-validation",
            "overfitting", "underfitting", "precision", "recall", "f1", "accuracy"
        ]
        
        response_lower = response.lower()
        for term in key_data_science_terms:
            if term in response_lower:
                topics.append(term)
        
        # Extract phrases after topic indicators
        for phrase in ["talked about", "discussed", "mentioned", "regarding", "about"]:
            if phrase in response_lower:
                parts = response_lower.split(phrase)
                if len(parts) > 1:
                    topic = parts[1].split(".")[0].strip()
                    if 3 < len(topic) < 30:  # Reasonable length for a topic
                        topics.append(topic)
        
        return topics[:3]  # Limit to top 3 topics to avoid over-constraining

    def is_positive_feedback(self, feedback):
        """Analyzes feedback to determine if it's positive."""
        # Use a simple heuristic based on sentiment words
        positive_indicators = [
            "excellent", "great", "good", "well", "strong", "clear", "thorough", 
            "impressive", "insightful", "solid", "detailed", "comprehensive"
        ]
        
        negative_indicators = [
            "could", "should", "consider", "try", "might want to", "would benefit", 
            "lacks", "missing", "unclear", "vague", "insufficient", "needs", "improve"
        ]
        
        feedback_lower = feedback.lower()
        
        # Count positive and negative indicators
        positive_count = sum(1 for word in positive_indicators if word in feedback_lower)
        negative_count = sum(1 for word in negative_indicators if word in feedback_lower)
        
        # Simple rule: if there are more negative than positive indicators, it's not positive
        return positive_count > negative_count

    async def determine_experience_level(self, introduction):
        """Analyzes the introduction to determine the candidate's experience level."""
        # Create a simple prompt to determine experience level
        prompt = f"""
        Based on the candidate's introduction below, determine their experience level in data science.
        
        Introduction: {introduction}
        
        Consider:
        - Years of experience mentioned
        - Types of projects/roles mentioned
        - Level of technical terminology used
        - Academic background
        
        Respond with ONLY one of these categories:
        - "junior" (0-2 years of experience)
        - "intermediate" (3-5 years of experience)
        - "senior" (5+ years of experience)
        """
        
        try:
            llm = initialize_llm(self.openai_api_key)
            response = await llm.ainvoke(prompt)
            experience_level = response.content.strip().lower()
            
            # Validate the response
            if experience_level not in ["junior", "intermediate", "senior"]:
                
                experience_level = "junior"
            print(f"Experience level determined: {experience_level}")    
            # Initialize difficulty level
            self.current_difficulty = "basic"

        # Adjust difficulty based on user experience
            if self.user_experience == "senior":
                self.current_difficulty = "intermediate"
            return experience_level
        except Exception as e:
            self.current_difficulty = "basic"
            return "junior"  # Default to junior in case of error

    async def run_interview(self,job_title, experience_text):
        """Runs the entire interview process asynchronously."""
        
        await self.get_job_description(job_title)
        await self.get_user_experience(experience_text)
            
            # Initialize RAG engine after getting job description and user experience
        await self.initialize_rag_engine()
        print(self,self.job_description,self.user_experience)
        return (self, self.job_description,self.user_experience)
        """await self.conduct_interview()
        except Exception as e:
    
            print(f"Error during interview: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # Show a friendly message to the user
            error_message = "I apologize, but we've encountered a technical issue with the interview process. Your responses have been saved, but we need to end the session now."
            print(f"AI: {error_message}")
            await self.speak_text_with_polly(error_message)
            
            # Return gracefully
            return False
        
        return True"""

    async def record_and_process_answer(self, audio_file_path):
        """Processes the user's recorded answer."""
        try:
            # Transcribe the audio using Whisper or another transcription service
            user_response = await transcribe_audio_with_whisper_async(audio_file_path, self.openai_api_key)
            if not user_response:
                return None, "Could not understand your response."

            # Analyze the response (optional)
            if self.asked_questions:
                feedback = await self.response_analyzer.analyze_response(
                    user_response, self.asked_questions[-1], self.openai_api_key
                )
            else:
                feedback = "No previous question available for analysis."

            print(f"\nUser Response: {user_response}")
            print(f"Feedback: {feedback}")

            # Add the response and feedback to the memory
            self.answers.append(user_response)
            memory.chat_memory.add_ai_message(self.asked_questions[-1])
            memory.chat_memory.add_user_message(user_response)
            add_analysis_to_history(memory, feedback)

            return user_response, feedback
        except Exception as e:
            print(f"Error processing answer: {e}", file=sys.stderr)
            return None, "An error occurred while processing the answer."
