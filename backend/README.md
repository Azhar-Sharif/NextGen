# AI-Powered Interview System

An automated AI interviewer for data science positions. This system conducts comprehensive technical interviews, analyzes candidate responses in real-time, and generates detailed feedback reports.

## Features

- **Interactive Voice Interface**: Speaks questions and records candidate responses
- **Contextual Question Generation**: Creates questions tailored to the job description and candidate experience
- **RAG-Enhanced Technical Evaluation**: Uses retrieval-augmented generation to evaluate technical answers against expert knowledge
- **Adaptive Difficulty**: Adjusts question difficulty based on candidate performance
- **Comprehensive Feedback**: Generates detailed PDF reports with scores and improvement suggestions
- **Multi-Modal Analysis**: Analyzes responses using NLP techniques for sentiment, key terms, and clarity

## Architecture

The system consists of several core modules:

- **Interview Flow**: Manages the overall interview process
- **Question Generation**: Creates tailored interview questions by category
- **Audio Processing**: Handles voice interaction (recording and text-to-speech)
- **RAG Engine**: Retrieves relevant technical content for evaluation
- **Response Analysis**: Analyzes candidate responses using various NLP techniques
- **Feedback Generation**: Creates comprehensive assessment reports

## Requirements

- Python 3.8+
- OpenAI API key
- AWS account for Amazon Polly (for voice synthesis)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-interview-system.git
   cd ai-interview-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install spaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Create a `config.ini` file with your API keys:
   ```ini
   [openai]
   api_key = your_openai_api_key_here

   [aws]
   region_name = your_aws_region_here
   polly_voice_id = Joanna
   ```

## Usage

Run the interview system:

```bash
python main.py
```

### Command Line Options

- `--config PATH`: Specify a custom config file path
- `--debug`: Enable debug mode with additional logging

## Development

### Project Structure

```
├── interview_project/
│   ├── __init__.py
│   ├── audio_processing.py      # Audio recording and transcription
│   ├── feedback_generation.py   # Interview feedback generation
│   ├── interview_flow.py        # Main interview process
│   ├── question_generation.py   # Interview question generation
│   ├── rag_engine.py            # RAG for technical evaluation
│   ├── response_analysis.py     # Response analysis with NLP
│   └── utils.py                 # Utility functions
├── data/                        # Data storage directory
├── reports/                     # Generated feedback reports
├── config.ini                   # Configuration file
├── main.py                      # Entry point
└── requirements.txt             # Dependencies
```

### Adding New Question Types

To add a new question type:

1. Add a new template in `question_generation.py`
2. Update the `conduct_interview` method in `interview_flow.py`

## License

[MIT License](LICENSE)

## Acknowledgments

- OpenAI API for powerful language processing
- Amazon Web Services for Polly text-to-speech
- HuggingFace Transformers for NLP capabilities
