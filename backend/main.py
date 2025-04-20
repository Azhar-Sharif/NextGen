#!/usr/bin/env python3
"""
AI-powered Interview System

This application runs an automated interview system that conducts data science interviews,
asks relevant questions, and generates feedback based on the candidate's responses.
"""

import asyncio
import argparse
import sys
import os
from dotenv import load_dotenv
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.interview_project.interview_flow import *
# Prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def main(job_title, experience_text, config_file=None, debug=False):
    """
    Main function to run the interview process.
    """
    load_dotenv()
    interviewer = Interviewer(config_file=config_file, debug=debug)
    # Pass job_title and experience_text to run_interview
    interview_instance,job_title,experience_text = await interviewer.run_interview(job_title, experience_text)
    print(f"Job Title: {job_title} Experience Text: {experience_text} interview_instanc:{interview_instance}")
    return (interview_instance,job_title, experience_text)
if __name__ == "__main__":
    """# Set up command line arguments
    parser = argparse.ArgumentParser(description="Run an AI-powered Interview")
    parser.add_argument("--job_title", required=True, help="Job title for the interview")
    parser.add_argument("--experience_text", required=True, help="Candidate's experience text")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    """
    # Run the main function and exit with its return code
    sys.exit(asyncio.run(main(job_title=args.job_title, experience_text=args.experience_text, config_file=args.config, debug=args.debug)))