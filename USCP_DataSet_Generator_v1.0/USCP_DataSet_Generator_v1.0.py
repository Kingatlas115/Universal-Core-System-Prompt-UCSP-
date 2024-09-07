import csv
import time
import random
import datetime
import hashlib
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import argparse
import sys
import traceback
import os
from tqdm import tqdm
import re

# Default values
DEFAULT_OPENAI_MODEL = "gpt-4"
DEFAULT_OPENROUTER_MODEL = "nousresearch/nous-hermes-llama2-13b"
DEFAULT_OUTPUT_CSV = "Universal_Core_System_Prompt_training_data.csv"
DEFAULT_NUM_TOOL = 60
DEFAULT_NUM_WORKFLOW_MANAGEMENT = 40
DEFAULT_NUM_DEBUG = 50
DEFAULT_NUM_CODE_GENERATION = 120
DEFAULT_NUM_OPTIMIZATION = 50
DEFAULT_NUM_VISUALIZATION = 50
DEFAULT_NUM_TRANSLATION = 40
DEFAULT_NUM_MATH = 40
DEFAULT_NUM_METADATA = 50
DEFAULT_NUM_COMBINED_FEATURES = 150

# Load the UCSP configuration
with open('ucsp_config.json', 'r') as f:
    UNIVERSAL_CORE_SYSTEM_PROMPT = json.load(f)

# Set to store prompt hashes
prompt_hashes = set()

def generate_hash(text):
    """Generate a hash for the given text."""
    return hashlib.md5(text.encode()).hexdigest()

def is_duplicate_prompt(prompt):
    """Check if a prompt is a duplicate based on its hash."""
    prompt_hash = generate_hash(prompt)
    if prompt_hash in prompt_hashes:
        return True
    prompt_hashes.add(prompt_hash)
    return False

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a dataset for the Universal Core System Prompt")
    parser.add_argument("--openai_api_key", help="OpenAI API Key")
    parser.add_argument("--openai_model", default=DEFAULT_OPENAI_MODEL, help="OpenAI Model to use")
    parser.add_argument("--openrouter_api_key", help="OpenRouter API Key")
    parser.add_argument("--openrouter_model", default=DEFAULT_OPENROUTER_MODEL, help="OpenRouter Model to use")
    parser.add_argument("--prompt_generation_model", help="Model to use for prompt generation")
    parser.add_argument("--response_generation_model", help="Model to use for response generation")
    parser.add_argument("--grading_model", help="Model to use for grading")
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV file name")
    parser.add_argument("--num_tool", type=int, default=DEFAULT_NUM_TOOL, help="Number of Tool examples")
    parser.add_argument("--num_workflow_management", type=int, default=DEFAULT_NUM_WORKFLOW_MANAGEMENT, help="Number of WorkflowManagement examples")
    parser.add_argument("--num_debug", type=int, default=DEFAULT_NUM_DEBUG, help="Number of Debug examples")
    parser.add_argument("--num_code_generation", type=int, default=DEFAULT_NUM_CODE_GENERATION, help="Number of CodeGeneration examples")
    parser.add_argument("--num_optimization", type=int, default=DEFAULT_NUM_OPTIMIZATION, help="Number of Optimization examples")
    parser.add_argument("--num_visualization", type=int, default=DEFAULT_NUM_VISUALIZATION, help="Number of Visualization examples")
    parser.add_argument("--num_translation", type=int, default=DEFAULT_NUM_TRANSLATION, help="Number of Translation examples")
    parser.add_argument("--num_math", type=int, default=DEFAULT_NUM_MATH, help="Number of Math examples")
    parser.add_argument("--num_metadata", type=int, default=DEFAULT_NUM_METADATA, help="Number of Metadata examples")
    parser.add_argument("--num_combined_features", type=int, default=DEFAULT_NUM_COMBINED_FEATURES, help="Number of CombinedFeatures examples")
    return parser.parse_args()

def generate_scenario_prompt(feature):
    """Generate a scenario prompt for the given feature."""
    templates = {
        "Tool": "Create a detailed real-world scenario where a tool integration is needed to {action} {data_type} from {source} for {purpose} in the {industry} industry.",
        "WorkflowManagement": "Design a complex workflow that {action} {data_type} from {source}, {operation}, and {result} for a {industry} company.",
        "Debug": "Describe a situation where a developer needs to identify and fix issues in a {complexity} {language} application that {task} in the {industry} sector.",
        "CodeGeneration": "Present a scenario where a software engineer needs to create a {complexity} {language} program that {task} for a {industry} project.",
        "Optimization": "Outline a case where performance optimization is required for a {complexity} {language} system that {task} in a {industry} environment.",
        "Visualization": "Describe a situation where data visualization is needed to represent {complexity} {data_type} for {purpose} in the {industry} field.",
        "Translation": "Present a scenario where language translation is needed for {content_type} from {source_language} to {target_language} for a {industry} organization.",
        "Math": "Create a detailed problem statement that requires solving a {complexity} {math_field} problem involving {concept} for a {industry} application.",
        "Metadata": "Describe a scenario where generating and using metadata is crucial for {purpose} in a {complexity} {industry} application.",
        "CombinedFeatures": "Design an advanced scenario that requires the use of multiple features to {task} and {result} for a complex {industry} project."
    }
    
    template = templates[feature]
    return template.format(
        action=random.choice(["extract", "process", "analyze", "transform"]),
        data_type=random.choice(["customer data", "financial records", "sensor readings", "social media posts"]),
        source=random.choice(["a REST API", "a database", "IoT devices", "web scraping"]),
        purpose=random.choice(["improve decision-making", "optimize operations", "enhance customer experience", "predict market trends"]),
        operation=random.choice(["perform sentiment analysis", "generate insights", "create visualizations", "trigger alerts"]),
        industry=random.choice(["healthcare", "finance", "e-commerce", "manufacturing", "education", "energy", "transportation"]),
        result=random.choice(["store in a data warehouse", "send email notifications", "update dashboards", "generate reports"]),
        language=random.choice(["Python", "JavaScript", "Java", "C++", "Go", "Rust"]),
        task=random.choice(["processes user input", "handles data transactions", "optimizes resource allocation", "implements a machine learning model"]),
        complexity=random.choice(["simple", "moderate", "complex", "highly sophisticated"]),
        content_type=random.choice(["technical documentation", "marketing material", "user interface", "legal contracts"]),
        source_language=random.choice(["English", "Spanish", "Mandarin", "German", "French"]),
        target_language=random.choice(["Japanese", "Korean", "Arabic", "Russian", "Portuguese"]),
        math_field=random.choice(["calculus", "linear algebra", "statistics", "differential equations", "optimization"]),
        concept=random.choice(["integration", "matrix operations", "probability distributions", "partial derivatives", "linear programming"])
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_prompt_and_response(feature, config):
    """Generate a prompt and response for the given feature using the LLM."""
    try:
        scenario_prompt = generate_scenario_prompt(feature)
        
        system_message = f"""You are an AI assistant with the capabilities described in the following system prompt:

{json.dumps(UNIVERSAL_CORE_SYSTEM_PROMPT, indent=2)}

Your task is to generate a realistic user prompt based on the given scenario, and then provide a detailed response demonstrating the proper use of the feature mentioned in the scenario. The user prompt should naturally incorporate the scenario without explicitly mentioning the feature.

Your response must strictly adhere to all guidelines and specifications in the UCSP. Pay particular attention to:
1. Correct use of tools and APIs as defined in the UCSP
2. Proper generation and use of metadata using the appropriate tags (<Tool>, <WorkflowManagement>, <Debug>, <CodeGeneration>, <Optimization>, <Visualization>, <Translation>, <Math>, <AssistantThinking>)
3. Appropriate creation and use of artifacts using the <Artifact> tag
4. Maintaining the specified interaction guidelines
5. Respecting the capabilities and limitations outlined in the UCSP

IMPORTANT: Do not start your responses with words or phrases such as 'Certainly,' 'Of course,' 'Absolutely,' 'Great,' 'Sure,' or similar expressions. Begin each response by immediately addressing the user's query or request.

When creating artifacts, follow these steps:
1. Use <AssistantThinking> tags to evaluate if the content qualifies as an artifact (this is considered metadata).
2. Wrap the artifact content in opening and closing <Artifact> tags.
3. Include the following attributes in the opening <Artifact> tag:
   - identifier: A unique, descriptive identifier in kebab-case
   - type: The appropriate MIME type for the content (e.g., "application/vnd.llama.code" for code)
   - title: A brief title or description of the content
   - language: For code artifacts, specify the programming language
4. Place the content between the opening and closing tags without using triple backticks.

Please provide your output in the following format:
Prompt: [Your generated user prompt]
Response: [Your detailed response as the AI assistant, strictly following the UCSP guidelines without visible tags or unnecessary introductory phrases]
[Include any metadata tags (AssistantThinking, Tool, WorkflowManagement, etc.) and Artifact tags as appropriate]
"""

        client = config['openai_client'] if config['prompt_generation_model'].startswith("gpt-") else config['openrouter_client']
        if client is None:
            raise ValueError("No valid client available for prompt generation")
        
        completion = client.chat.completions.create(
            model=config['prompt_generation_model'],
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate a prompt and response based on the given scenario: {scenario_prompt}"}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        
        if not completion.choices:
            raise ValueError("No choices returned from the API")
        
        result = completion.choices[0].message.content.strip()
        if not result:
            raise ValueError("Empty result returned from the API")
        
        # Parse the result
        prompt, response, metadata, artifacts = parse_result(result)
        
        # Generate response using a different model if specified
        if config['response_generation_model'] and config['response_generation_model'] != config['prompt_generation_model']:
            client = config['openai_client'] if config['response_generation_model'].startswith("gpt-") else config['openrouter_client']
            if client is None:
                raise ValueError("No valid client available for response generation")
            
            response_completion = client.chat.completions.create(
                model=config['response_generation_model'],
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            if not response_completion.choices:
                raise ValueError("No choices returned from the API for response generation")
            
            response_result = response_completion.choices[0].message.content.strip()
            if not response_result:
                raise ValueError("Empty result returned from the API for response generation")
            
            _, new_response, new_metadata, new_artifacts = parse_result(response_result)
            if new_response:
                response = new_response
            if new_metadata:
                metadata = new_metadata
            if new_artifacts:
                artifacts = new_artifacts
        
        if is_duplicate_prompt(prompt):
            raise ValueError("Duplicate prompt generated")
        
        return prompt, response, metadata, artifacts
    except Exception as e:
        print(f"Error in generate_prompt_and_response for feature {feature}: {str(e)}")
        raise

def parse_result(result):
    """Parse the result from the LLM to extract prompt, response, metadata, and artifacts."""
    parts = result.split("\n")
    prompt = ""
    response = ""
    metadata = []
    artifacts = []
    current_section = None
    current_content = ""

    metadata_tags = ['Tool', 'WorkflowManagement', 'Debug', 'CodeGeneration', 'Optimization', 'Visualization', 'Translation', 'Math', 'AssistantThinking']

    for part in parts:
        if part.startswith("Prompt:"):
            current_section = "prompt"
            prompt = part.replace("Prompt:", "").strip()
        elif part.startswith("Response:"):
            current_section = "response"
            response = part.replace("Response:", "").strip()
        elif any(part.startswith(f"<{tag}") for tag in metadata_tags):
            if current_section == "metadata":
                metadata.append(current_content.strip())
            current_section = "metadata"
            current_content = part
        elif part.startswith("</"):
            if current_section == "metadata":
                current_content += "\n" + part
                metadata.append(current_content.strip())
                current_content = ""
            elif current_section == "artifact":
                current_content += "\n" + part
                artifacts.append(current_content.strip())
                current_content = ""
            current_section = None
        elif part.startswith("<Artifact"):
            if current_section == "artifact":
                artifacts.append(current_content.strip())
            current_section = "artifact"
            current_content = part
        elif current_section == "response":
            response += "\n" + part
        elif current_section in ["metadata", "artifact"]:
            current_content += "\n" + part

    # Add any remaining content
    if current_section == "metadata" and current_content:
        metadata.append(current_content.strip())
    elif current_section == "artifact" and current_content:
        artifacts.append(current_content.strip())

    return prompt, response, metadata, artifacts

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_for_grading(prompt, response, metadata, artifacts, feature, config):
    """Call the API to grade the generated prompt-response pair with focus on correctness and feature-specific criteria."""
    grading_criteria = {
        "Tool": "Evaluate the correct implementation and effective use of tool integration.",
        "WorkflowManagement": "Assess the correctness and effectiveness of the workflow management solution.",
        "Debug": "Evaluate the accuracy and effectiveness of the debugging approach.",
        "CodeGeneration": "Assess the correctness, efficiency, and appropriateness of the generated code.",
        "Optimization": "Evaluate the correctness and effectiveness of the optimization techniques used.",
        "Visualization": "Assess the accuracy and effectiveness of the data visualization approach.",
        "Translation": "Evaluate the accuracy and appropriateness of the language translation.",
        "Math": "Assess the correctness and efficiency of the mathematical problem-solving approach.",
        "Metadata": "Evaluate the accuracy and appropriate use of generated metadata.",
        "CombinedFeatures": "Evaluate the correct and effective use of multiple features in solving the given problem."
    }

    grading_prompt = f"""
    Evaluate the following prompt, response, metadata, and artifacts. Your primary focus should be on the correctness and accuracy of the response in relation to the prompt. Consider the following criteria:
    
    1. Correctness: Assess whether the response accurately and effectively addresses the prompt. This is the most important factor.
    2. Accuracy: Evaluate the factual correctness of any information provided in the response.
    3. Effectiveness: Determine how well the response solves the problem or fulfills the request in the prompt.
    4. {grading_criteria.get(feature, "Provide a general assessment of the quality and relevance of the response.")}
    
    Secondary considerations:
    - Adherence to the UCSP guidelines
    - Proper use of tools and APIs as defined in the UCSP
    - Appropriate use of metadata and artifacts
    - Clarity and specificity in the explanation or implementation
    - Practical relevance and potential usefulness of the solution

    Prompt: {prompt}
    Response: {response}
    Metadata: {metadata}
    Artifacts: {artifacts}

    Provide your evaluation in the following format:
    Score: [Your score from 1 to 10, where 1 is poor and 10 is excellent]
    Explanation: [Your detailed explanation, focusing primarily on the correctness and effectiveness of the response in addressing the prompt]
    """

    try:
        client = config['openai_client'] if config['grading_model'].startswith("gpt-") else config['openrouter_client']
        if client is None:
            raise ValueError("No valid client available for grading")
        
        completion = client.chat.completions.create(
            model=config['grading_model'],
            messages=[
                {"role": "system", "content": "You are an expert AI assistant tasked with grading responses based on their correctness, accuracy, and effectiveness in addressing the given prompt."},
                {"role": "user", "content": grading_prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        if not completion.choices:
            raise ValueError("No choices returned from the API for grading")
        
        result_text = completion.choices[0].message.content.strip()
        if not result_text:
            raise ValueError("Empty result returned from the API for grading")
        
        score, explanation = parse_grading_result(result_text)
        return score, explanation
    except Exception as e:
        print(f"Error calling API for grading: {str(e)}")
        raise

def parse_grading_result(result_text):
    """Parse the grading result text to extract the score and explanation."""
    try:
        lines = result_text.split('\n')
        score = None
        explanation = []
        for line in lines:
            if line.startswith('Score:'):
                score = int(line.split(':')[1].strip())
            elif line.startswith('Explanation:'):
                explanation = [line.split(':', 1)[1].strip()]
            elif score is not None:
                explanation.append(line.strip())
        
        if score is None:
            raise ValueError("No score found in the grading result.")
        
        explanation = ' '.join(explanation).strip()
        return score, explanation
    except Exception as e:
        print(f"Error parsing grading result: {e}")
        print(f"Full result text: {result_text}")
        return 0, "Failed to parse grading result."

def save_to_csv(file_path, data):
    """Save a single row of data to the CSV file."""
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            metadata_str = json.dumps(data['metadata']) if isinstance(data['metadata'], list) else str(data['metadata'])
            artifacts_str = json.dumps(data['artifacts']) if isinstance(data['artifacts'], list) else str(data['artifacts'])
            writer.writerow([
                data['prompt'],
                data['response'],
                metadata_str,
                artifacts_str,
                data['score'],
                data['explanation'],
                data['feature']
            ])
        print(f"Saved data to CSV: {data['feature']} (Score: {data['score']})")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        print(f"Data being saved: {data}")

def generate_and_grade_examples(config):
    """Main function to generate prompt-response pairs and grade them."""
    features = {
        "Tool": config['num_tool'],
        "WorkflowManagement": config['num_workflow_management'],
        "Debug": config['num_debug'],
        "CodeGeneration": config['num_code_generation'],
        "Optimization": config['num_optimization'],
        "Visualization": config['num_visualization'],
        "Translation": config['num_translation'],
        "Math": config['num_math'],
        "Metadata": config['num_metadata'],
        "CombinedFeatures": config['num_combined_features']
    }

    total_examples = sum(features.values())
    
    with tqdm(total=total_examples, desc="Generating Dataset", unit="example") as pbar:
        for feature, num_examples in features.items():
            for _ in range(num_examples):
                try:
                    tqdm.write(f"Generating Dataset for {feature}:")
                    prompt, response, metadata, artifacts = generate_prompt_and_response(feature, config)
                    tqdm.write(f'Generated Prompt: "{prompt}"')
                    tqdm.write("Response, metadata, and artifacts generated. Grading...")
                    score, explanation = call_for_grading(prompt, response, metadata, artifacts, feature, config)
                    save_to_csv(config['output_csv'], {
                        'prompt': prompt,
                        'response': response,
                        'metadata': metadata,
                        'artifacts': artifacts,
                        'score': score,
                        'explanation': explanation,
                        'feature': feature
                    })
                    tqdm.write(f"Example graded. Score: {score}")
                    pbar.update(1)
                    tqdm.write(f"DATA:{prompt[:50]}...,{response[:50]}...,{str(metadata)[:50]}...,{str(artifacts)[:50]}...,{score},{feature}")
                except Exception as e:
                    tqdm.write(f"Error generating or grading example for {feature}: {str(e)}")
                    tqdm.write(f"Full traceback: {traceback.format_exc()}")
                sys.stdout.flush()  # Ensure output is sent immediately
                time.sleep(2)  # Pause to avoid hitting rate limits

def main():
    try:
        print("Script started")
        args = parse_arguments()
        
        config = {
            'openai_api_key': os.environ.get('OPENAI_API_KEY') or args.openai_api_key,
            'openrouter_api_key': os.environ.get('OPENROUTER_API_KEY') or args.openrouter_api_key,
            'openai_model': args.openai_model,
            'openrouter_model': args.openrouter_model,
            'prompt_generation_model': args.prompt_generation_model or args.openrouter_model,
            'response_generation_model': args.response_generation_model or args.openrouter_model,
            'grading_model': args.grading_model or args.openai_model,
            'output_csv': args.output_csv,
            'num_tool': args.num_tool,
            'num_workflow_management': args.num_workflow_management,
            'num_debug': args.num_debug,
            'num_code_generation': args.num_code_generation,
            'num_optimization': args.num_optimization,
            'num_visualization': args.num_visualization,
            'num_translation': args.num_translation,
            'num_math': args.num_math,
            'num_metadata': args.num_metadata,
            'num_combined_features': args.num_combined_features
        }

        # Initialize OpenAI client if OpenAI API key is provided
        if config['openai_api_key']:
            config['openai_client'] = OpenAI(api_key=config['openai_api_key'])
            print("OpenAI client initialized")
        else:
            config['openai_client'] = None
            print("OpenAI client not initialized (no API key provided)")
        
        # Initialize OpenRouter client if OpenRouter API key is provided
        if config['openrouter_api_key']:
            config['openrouter_client'] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=config['openrouter_api_key'],
                default_headers={
                    "HTTP-Referer": "https://github.com/DanKulik",
                    "X-Title": "UCSP Dataset Generator"
                }
            )
            print("OpenRouter client initialized")
        else:
            config['openrouter_client'] = None
            print("OpenRouter client not initialized (no API key provided)")

        # Prepare CSV file with headers
        with open(config['output_csv'], mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['prompt', 'response', 'metadata', 'artifacts', 'score', 'explanation', 'feature'])
        print(f"CSV file '{config['output_csv']}' prepared with headers")

        # Start generating and grading examples
        print("Starting to generate and grade examples...")
        generate_and_grade_examples(config)

        print(f"Dataset generation complete. Output saved to {config['output_csv']}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()