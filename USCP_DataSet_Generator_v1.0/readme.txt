UCSP Dataset Generation Script
==============================

Overview:
---------
This script is designed to generate a comprehensive dataset for the Universal Core System Prompt (UCSP) project. It creates prompts, responses, metadata, and artifacts that adhere to the UCSP guidelines across various features and scenarios.

Key Features:
-------------
1. Multi-feature Support:
   - The script generates examples for multiple features including Tool Integration, Workflow Management, Debugging, Code Generation, Optimization, Visualization, Translation, Mathematical Problem-solving, Metadata Usage, and Combined Features.

2. Customizable Generation:
   - Users can specify the number of examples to generate for each feature via command-line arguments.

3. Model Flexibility:
   - Supports both OpenAI and OpenRouter models for content generation and grading.
   - Allows separate models for prompt generation, response generation, and grading.

4. Scenario Generation:
   - Dynamically creates diverse scenarios for each feature using templates and random selections.

5. Prompt and Response Generation:
   - Generates user prompts based on scenarios and produces detailed responses adhering to UCSP guidelines.

6. Metadata and Artifact Handling:
   - Creates and manages metadata and artifacts as per UCSP specifications.
   - Ensures proper JSON formatting for metadata and artifacts.

7. Grading System:
   - Implements an automated grading system to evaluate the quality of generated content.
   - Provides scores and explanations for each generated example.

8. Duplicate Prevention:
   - Implements a hashing mechanism to prevent duplicate prompts.

9. Error Handling and Retry Mechanism:
   - Utilizes retry decorators to handle temporary failures in API calls.
   - Comprehensive error logging for troubleshooting.

10. Progress Tracking:
    - Displays real-time progress information during dataset generation.

11. CSV Output:
    - Saves all generated data, including prompts, responses, metadata, artifacts, scores, and explanations, to a CSV file.

Usage:
------
The script is run from the command line with various arguments to customize the dataset generation process. Key arguments include:

- API keys for OpenAI and/or OpenRouter
- Model selections for different generation tasks
- Number of examples to generate for each feature
- Output CSV file name

Example command:
python script_name.py --openai_api_key YOUR_KEY --openai_model gpt-4 --num_tool 50 --output_csv ucsp_dataset.csv

Script Structure:
-----------------
1. Argument Parsing: Handles command-line arguments for customization.
2. Scenario Generation: Creates diverse scenarios for each feature.
3. Prompt and Response Generation: Generates prompts and responses using specified LLM.
4. Metadata and Artifact Handling: Creates and manages metadata and artifacts.
5. Grading System: Evaluates generated content for quality and adherence to UCSP.
6. CSV Output: Saves all generated data to a CSV file.

Configuration:
--------------
The script uses a 'ucsp_config.json' file to load the Universal Core System Prompt configuration. Ensure this file is present in the same directory as the script.

Dependencies:
-------------
- OpenAI Python Library
- Tenacity (for retry mechanism)
- Standard Python libraries: csv, json, random, datetime, hashlib, argparse, sys, traceback, os

Note:
-----
This script is designed to be run in an environment with access to the specified AI models and appropriate API keys. Ensure all necessary credentials and permissions are in place before running the script.

For any issues or further customization needs, please refer to the script comments or contact the development team.
