import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Client Simulator Email Credentials
SIMULATOR_EMAIL_ADDRESS = os.getenv("SIMULATOR_EMAIL_ADDRESS")

# Target Platform Email
PLATFORM_EMAIL_ADDRESS = os.getenv("PLATFORM_EMAIL_ADDRESS")

# Company Information
COMPANY_INFO = {
    "name": "ChamomTravel",
    "country": "Georgia",
    "services": ["tour planning", "hotel booking", "transportation"],
    "currencies": ["GEL", "USD", "EUR"]
}

# LLM Configuration
LLM_MODEL_NAME = "gpt-4o"  # "gpt-4o", "gpt-4", "gemini-pro"
LLM_TEMPERATURE = 0.7


# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API keys are set if specific LLMs are chosen
if "gpt" in LLM_MODEL_NAME and not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
elif "gemini" in LLM_MODEL_NAME and not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")


# Simulation Parameters
MAX_CONVERSATION_TURNS = 15
MIN_RESPONSE_DELAY_SECONDS = 60
MAX_RESPONSE_DELAY_SECONDS = 120
ADD_RANDOM_FOLLOW_UP_CHANCE = 0.2

# Analytics & Logging
LOG_FILE_PATH = "logs/simulation_log.json"