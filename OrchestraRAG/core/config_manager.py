# core/config_manager.py
import tomli
import os
import sys
import logging

from dotenv import load_dotenv
load_dotenv() # This loads variables from .env into os.environ


logger = logging.getLogger(__name__)

class AppConfig:
    _instance = None
    _config_data: dict = {}

    def __new__(cls):
        # Implement Singleton pattern to ensure only one config instance exists.
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        config_dir_name = "config"
        config_file_name = "settings.toml"

        # Prioritize loading config from 'config/' within the current working directory.
        config_path = os.path.join(os.getcwd(), config_dir_name, config_file_name)

        if not os.path.exists(config_path):
            # If not found in CWD, try looking in 'config/' within the script's directory.
            # This accounts for running from different locations (e.g., `python api/main.py` vs `uvicorn api.main:app`).
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root, then into 'config'
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, config_dir_name, config_file_name)
        
        # If the configuration file is still not found, log an error and exit.
        if not os.path.exists(config_path):
            logger.error(f"Configuration file '{config_file_name}' not found at expected locations: "
                         f"'{os.path.join(os.getcwd(), config_dir_name)}/{config_file_name}' or "
                         f"'{os.path.join(project_root, config_dir_name)}/{config_file_name}'. Exiting.")
            sys.exit(1) # Critical error: application cannot run without configuration.

        try:
            # Open the TOML file in binary read mode for tomli.
            with open(config_path, "rb") as f:
                self._config_data = tomli.load(f)
            logger.info(f"Configuration loaded successfully from: {config_path}")
        except tomli.TOMLDecodeError as e:
            # Handle errors specific to TOML parsing.
            logger.error(f"Error decoding TOML configuration from {config_path}: {e}. Exiting.")
            sys.exit(1)
        except Exception as e:
            # Catch any other unexpected errors during file loading.
            logger.error(f"An unexpected error occurred while loading config from {config_path}: {e}. Exiting.")
            sys.exit(1)

    def get(self, section: str, key: str, default=None):
        """
        Safely retrieves a configuration value from a specified section and key.
        
        Args:
            section (str): The name of the section in the TOML file (e.g., "rag_service").
            key (str): The key within that section (e.g., "base_url").
            default: An optional default value to return if the section or key is not found.

        Returns:
            The configuration value, or the default if not found. Logs a warning if no default
            is provided and the key is missing.
        """
        value = self._config_data.get(section, {}).get(key, default)
        if value is None and default is None:
            # Warn if a key is accessed without a default and isn't found, which might indicate a misconfiguration.
            logger.warning(f"Configuration key '{key}' not found in section '{section}' and no default provided. Returning None.")
        return value

# Initialize the configuration globally within this module.
# Other modules will import `config` from `core.config_manager`.
config = AppConfig()

# Export specific config values for convenience
#BASE_URL = config.get("rag_service", "base_url", "http://localhost:8083")
BASE_URL = os.getenv("RAG_SERVICE_BASE_URL", config.get("rag_service", "base_url", "http://localhost:8083"))
MAX_RAG_RETRIES = config.get("rag_service", "max_rag_retries", 2)
USE_EMOJIS = config.get("app_settings", "use_emojis", True)
MIN_RELEVANCE_SCORE = config.get("evaluation", "min_relevance_score", 3)

CLEAR_HISTORY_TIMEOUT = config.get("rag_service", "clear_history_timeout_seconds", 130)
CHECK_CACHE_TIMEOUT = config.get("rag_service", "check_cache_timeout_seconds", 30)
RAG_INVOKE_TIMEOUT = config.get("rag_service", "rag_timeout_seconds", 360)
SUMMARIZE_TIMEOUT = config.get("rag_service", "summarize_timeout_seconds", 360)
EVALUATE_SINGLE_TIMEOUT = config.get("rag_service", "eval_timeout_seconds", 320)
EVALUATE_MULTI_TIMEOUT = config.get("rag_service", "eval_all_timeout_seconds", 760)
