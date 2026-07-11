"""Create LLM instances from config profiles."""

import os
from typing import Any


def create_llm_from_profile(profile: dict[str, Any]) -> Any:
    provider = profile["provider"]
    model = profile["model"]
    temperature = profile.get("temperature", 0.3)
    api_key_env = profile.get("api_key_env", "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, anthropic_api_key=api_key  # type: ignore[call-arg]
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)  # type: ignore[call-arg]
    elif provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        return ChatBedrockConverse(
            model_id=model,  # type: ignore[call-arg]
            temperature=temperature,
            region_name=profile.get("region", "us-east-1"),
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def create_llm(config: dict[str, Any]) -> Any:
    profile_name = config["llm"]["active_profile"]
    profile = config["llm"]["profiles"][profile_name]
    return create_llm_from_profile(profile)
