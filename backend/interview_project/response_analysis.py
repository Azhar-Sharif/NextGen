import asyncio
from openai import AsyncOpenAI
from transformers import pipeline
from keybert import KeyBERT
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
import re
import sys
import json

class ResponseAnalyzer:
    def __init__(self):
        try:
            # Initialize Hugging Face NER pipeline for named entity recognition
            self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
            # Initialize KeyBERT for key phrase extraction
            self.kw_model = KeyBERT()
        except Exception as e:
            print(f"Warning: Could not initialize all NLP components: {e}", file=sys.stderr)
            self.ner_pipeline = None
            self.kw_model = None
            
        # Initialize VADER for sentiment analysis
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Data science specific terminology
        self.ds_terms = [
            'regression', 'classification', 'clustering', 'neural network', 'deep learning',
            'decision tree', 'random forest', 'gradient boosting', 'xgboost', 'lstm', 'cnn',
            'scikit-learn', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'matplotlib',
            'feature engineering', 'feature selection', 'cross-validation', 'hyperparameter',
            'overfitting', 'underfitting', 'bias-variance', 'precision', 'recall', 'f1',
            'roc', 'auc', 'accuracy', 'confusion matrix', 'type i error', 'type ii error',
            'p-value', 'hypothesis testing', 'statistical significance', 'correlation',
            'causation', 'time series', 'forecasting', 'normalization', 'standardization',
            'dimensionality reduction', 'pca', 't-sne', 'umap', 'supervised learning',
            'unsupervised learning', 'reinforcement learning', 'nlp', 'computer vision',
            'recommender system', 'collaborative filtering', 'content-based filtering',
            'a/b testing', 'etl', 'data warehouse', 'data lake', 'big data', 'hadoop',
            'spark', 'sql', 'nosql', 'mongodb', 'database', 'data pipeline', 'batch processing',
            'stream processing', 'model deployment', 'mlops', 'data drift', 'concept drift',
            'model monitoring', 'explainability', 'interpretability', 'shap', 'lime'
        ]

    async def analyze_response(self, response_text: str, question: str, openai_api_key: str) -> str:
        """Analyze the response and generate feedback using OpenAI."""
        client = AsyncOpenAI(api_key=openai_api_key)

        # Basic text analysis
        word_count = len(response_text.split())
        
        # Skip complex analysis for very short responses
        if word_count < 5:
            # For very short responses, provide basic feedback
            try:
                quick_feedback_prompt = f"""
                The candidate gave a very short response: "{response_text}" 
                to the question: "{question}".
                
                Provide brief, constructive feedback (max 15 words) encouraging more detailed responses.
                """
                
                response = await client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert data science interviewer."},
                        {"role": "user", "content": quick_feedback_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=50
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Error generating quick feedback: {e}", file=sys.stderr)
                return "Please provide a more detailed response to showcase your knowledge."
        
        # Full analysis for substantive responses
        try:
            # Sentiment analysis
            sentiment_scores = self.sentiment_analyzer.polarity_scores(response_text)
            compound_score = sentiment_scores['compound']
            sentiment_label = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"
            
            # Named entities and key phrases
            entity_texts = []
            key_phrases = []
            
            # Only run these if the components initialized successfully
            if self.ner_pipeline:
                try:
                    entities = sorted(self.ner_pipeline(response_text), key=lambda x: x['score'], reverse=True)[:5]
                    entity_texts = [entity['word'] for entity in entities]
                except Exception as e:
                    print(f"Error in NER processing: {e}", file=sys.stderr)
            
            if self.kw_model:
                try:
                    keywords = self.kw_model.extract_keywords(
                        response_text,
                        keyphrase_ngram_range=(1, 2),
                        stop_words='english',
                        top_n=5
                    )
                    key_phrases = [kw[0] for kw in keywords]
                except Exception as e:
                    print(f"Error in keyword extraction: {e}", file=sys.stderr)
            
            # Detect data science terminology
            ds_term_count = 0
            found_ds_terms = []
            for term in self.ds_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', response_text.lower()):
                    ds_term_count += 1
                    found_ds_terms.append(term)
            
            # Limit to top 5 most relevant DS terms
            found_ds_terms = found_ds_terms[:5]
            
            # Check for quantitative statements (numbers, percentages)
            quantitative_pattern = r'\b\d+(?:\.\d+)?%?\b'
            quantitative_mentions = len(re.findall(quantitative_pattern, response_text))
            
            # Check for mentions of practical application or tools
            tools_pattern = r'\b(?:python|r|sql|tableau|power bi|excel|jupyter|pandas|numpy|sklearn|tensorflow|pytorch|spacy|nltk|matplotlib|seaborn|plotly)\b'
            tool_mentions = len(re.findall(tools_pattern, response_text.lower()))
            
            # Calculate clarity score using Flesch reading ease
            clarity_score = textstat.flesch_reading_ease(response_text)

            # Prepare analysis for structured JSON output
            analysis_data = {
                "word_count": word_count,
                "sentiment": sentiment_label,
                "sentiment_score": compound_score,
                "key_phrases": key_phrases,
                "named_entities": entity_texts,
                "data_science_terms": found_ds_terms,
                "data_science_term_count": ds_term_count,
                "quantitative_elements": quantitative_mentions,
                "tool_mentions": tool_mentions,
                "clarity_score": clarity_score,
                "question": question,
                "answer": response_text
            }
            
            # Convert to JSON string
            analysis_json = json.dumps(analysis_data)

            # Prepare analysis prompt for GPT with structured input
            prompt = f"""
            Analyze this data science interview response based on the following analysis data:
            
            {analysis_json}
            
            As a data science interviewer, evaluate the response focusing on:
            1. Technical accuracy: Are the concepts correctly explained?
            2. Depth of knowledge: Does the candidate show deep understanding rather than surface-level knowledge?
            3. Practical application: Did they mention real-world applications or experiences?
            4. Critical thinking: Did they analyze trade-offs or limitations?
            5. Communication: Was the explanation clear and structured?

            Provide one line of meaningful feedback that is:
            1. Specific to the data science content in the response
            2. Either appreciative (if technically sound) or constructively critical (if needs improvement)
            3. Professional and encouraging
            4. Maximum 20 words

            Return ONLY the feedback line, nothing else.
            """

            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert data science interviewer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating feedback: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return "Thank you for your response."

def add_analysis_to_history(memory, feedback: str):
    """Adds the analysis feedback to the conversation history."""
    memory.chat_memory.add_ai_message(f"Feedback: {feedback}")