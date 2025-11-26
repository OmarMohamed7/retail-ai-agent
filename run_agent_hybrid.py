"""
Main entry point for the application.
"""

from agent.config import AgentConfig


if __name__ == "__main__":
    # Load configuration from environment
    config = AgentConfig()

    # Print app name
    print(config.app_name)
