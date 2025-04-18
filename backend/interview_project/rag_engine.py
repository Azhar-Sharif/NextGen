import faiss
import numpy as np
import json
import os
import torch
import sys
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Any, Optional, Union
import asyncio

class RAGEngine:
    """
    RAG Engine for technical question generation and answer evaluation.
    Uses a FAISS index to retrieve relevant technical information.
    """
    
    def __init__(self, index_path: str, chunks_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG Engine.
        
        Args:
            index_path: Path to the FAISS index file
            chunks_path: Path to the JSON file containing the text chunks
            model_name: Name of the sentence transformer model to use
        """
        self.index_path = index_path
        self.chunks_path = chunks_path
        
        # Load chunks
        try:
            with open(chunks_path, 'r') as f:
                self.chunks = json.load(f)
        except FileNotFoundError:
            print(f"Error: Chunks file not found at {chunks_path}", file=sys.stderr)
            raise
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in chunks file at {chunks_path}", file=sys.stderr)
            raise
            
        # Load index
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            print(f"Error loading FAISS index from {index_path}: {e}", file=sys.stderr)
            raise
        
        # Set device
        self.device = self._determine_device()
        print(f"RAG Engine using device: {self.device}")
        
        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            print(f"Embedding model loaded on {self.model.device}")
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}", file=sys.stderr)
            raise
    
    def _determine_device(self) -> str:
        """
        Determine the best available device for computation.
        
        Returns:
            str: Device name ('mps', 'cuda', or 'cpu')
        """
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"  # CPU fallback
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query string to an embedding vector.
        
        Args:
            query: The query string to encode
            
        Returns:
            A numpy array containing the embedding
        """
        if not query or not isinstance(query, str):
            print("Warning: Empty or invalid query for encoding", file=sys.stderr)
            # Return zero vector with correct dimensions as fallback
            return np.zeros((1, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
            
        try:
            with torch.no_grad():
                embedding = self.model.encode(query)
                return np.array(embedding).reshape(1, -1).astype('float32')
        except Exception as e:
            print(f"Error encoding query: {e}", file=sys.stderr)
            # Return zero vector with correct dimensions as fallback
            return np.zeros((1, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the k most relevant chunks for a query.
        
        Args:
            query: The query string
            k: Number of results to retrieve
            
        Returns:
            A list of the k most relevant chunks
        """
        if not query:
            print("Warning: Empty query for retrieval", file=sys.stderr)
            return []
            
        try:
            # Encode the query
            query_embedding = self.encode_query(query)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, k)
            
            # Get the corresponding chunks
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    print(f"Warning: Index {idx} out of bounds", file=sys.stderr)
                    continue
                results.append({
                    "chunk": self.chunks[int(idx)],
                    "score": float(distances[0][i])
                })
            
            return results
        except Exception as e:
            print(f"Error during retrieval: {e}", file=sys.stderr)
            return []
    
    async def generate_technical_question(
        self, 
        job_description: str, 
        user_experience: str, 
        difficulty: str,
        previous_questions: List[str],
        openai_client: Any
    ) -> str:
        """
        Generate a technical question based on retrieved context.
        
        Args:
            job_description: The job description
            user_experience: The user's experience
            difficulty: The difficulty level (basic, intermediate, advanced)
            previous_questions: List of previously asked questions
            openai_client: The OpenAI client to use for generation
            
        Returns:
            A generated technical question
        """
        if not job_description or not openai_client:
            print("Error: Missing required parameters for question generation", file=sys.stderr)
            return "Unable to generate a technical question at this time."
            
        try:
            # Combine job and experience for retrieval
            combined_query = f"Technical data science job: {job_description}. Experience: {user_experience}"
            
            # Retrieve relevant technical content
            retrieved_context = self.retrieve(combined_query, k=3)
            
            # Format context for the prompt
            context_str = "\n---\n".join(
                [f"Score: {item['score']:.4f}\nContent: {item['chunk']['text']}" for item in retrieved_context]
            )
            
            # Format previous questions
            prev_questions_str = "\n".join([f"- {q}" for q in previous_questions])
            
            # Create the prompt
            difficulty_descriptions = {
                "basic": "fundamental concepts that entry-level data scientists should know",
                "intermediate": "concepts that require 1-3 years of practical experience",
                "advanced": "advanced concepts that senior data scientists with 3+ years of experience should understand"
            }
            
            difficulty_desc = difficulty_descriptions.get(difficulty.lower(), difficulty_descriptions["intermediate"])
            
            prompt = f"""
            You are an expert data science interviewer creating technical questions.
            
            Job Description: {job_description}
            
            Candidate Experience: {user_experience}
            
            Based on the following relevant technical content:
            
            {context_str}
            
            Generate a {difficulty.upper()} level technical question about data science that focuses on {difficulty_desc}.
            
            Previously asked questions (DO NOT REPEAT THESE):
            {prev_questions_str}
            
            The question should:
            1. Be relevant to the job description and candidate experience
            2. Test understanding of core data science concepts at the appropriate difficulty level
            3. Be specific and focused on one technical concept
            4. Be answerable without access to external resources
            5. NOT repeat any previously asked questions or concepts
            
            Format: Return ONLY the question text. No prefixes, labels, or explanations.
            """
            
            # Generate the question
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data science interview expert creating technical questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            question = response.choices[0].message.content.strip()
            
            # Clean up the question
            if question.startswith("Question:"):
                question = question[len("Question:"):].strip()
                
            return question
            
        except Exception as e:
            print(f"Error generating technical question: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return "Unable to generate a technical question at this time."
    
    async def evaluate_technical_answer(
        self, 
        question: str, 
        answer: str, 
        job_description: str,
        openai_client: Any
    ) -> Dict[str, Any]:
        """
        Evaluate a technical answer using RAG.
        
        Args:
            question: The technical question asked
            answer: The candidate's answer
            job_description: The job description
            openai_client: The OpenAI client
            
        Returns:
            A dictionary containing the evaluation results
        """
        if not question or not answer or not openai_client:
            print("Error: Missing required parameters for answer evaluation", file=sys.stderr)
            return {
                "score": 5,
                "feedback": "Could not properly evaluate the answer due to missing information."
            }
            
        try:
            # Retrieve relevant technical content for the question
            retrieved_context = self.retrieve(question, k=3)
            
            # Format context for the prompt
            context_str = "\n---\n".join(
                [f"Score: {item['score']:.4f}\nContent: {item['chunk']['text']}" for item in retrieved_context]
            )
            
            # Create the prompt
            prompt = f"""
            You are an expert data science interviewer evaluating a candidate's answer to a technical question.
            
            Question: {question}
            
            Candidate's Answer: {answer}
            
            Relevant technical context:
            {context_str}
            
            Job Description: {job_description}
            
            Evaluate the candidate's answer based on:
            1. Technical accuracy (is the answer correct according to the context?)
            2. Completeness (does it cover key aspects of the topic?)
            3. Clarity (is the explanation clear and well-structured?)
            4. Relevance (is it relevant to the job requirements?)
            
            Return your evaluation in the following JSON format:
            {{
                "score": <number from 1-10>,
                "feedback": "<specific feedback on the answer>",
                "areas_of_strength": ["<strength 1>", "<strength 2>"],
                "areas_for_improvement": ["<area 1>", "<area 2>"]
            }}
            
            Your response should ONLY contain valid JSON.
            """
            
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data science interview evaluator. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
            
        except json.JSONDecodeError as e:
            print(f"Error parsing evaluation JSON: {e}", file=sys.stderr)
            return {
                "score": 5,
                "feedback": "There was an error evaluating your answer. The system could not properly analyze your response."
            }
        except Exception as e:
            print(f"Error evaluating technical answer: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return {
                "score": 5,
                "feedback": "There was an error evaluating your answer. Please continue with the interview."
            }
    
    @classmethod
    async def create_engine(cls, index_path: str, chunks_path: str, model_name: str = 'all-MiniLM-L6-v2') -> 'RAGEngine':
        """
        Factory method to create a RAG engine asynchronously.
        
        Args:
            index_path: Path to the FAISS index file
            chunks_path: Path to the JSON file containing the text chunks
            model_name: Name of the sentence transformer model to use
            
        Returns:
            A configured RAGEngine instance
        """
        try:
            # Run model loading in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            engine = await loop.run_in_executor(
                None,
                lambda: cls(index_path, chunks_path, model_name)
            )
            return engine
        except Exception as e:
            print(f"Error creating RAG engine: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise 