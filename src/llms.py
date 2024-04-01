import anthropic
from openai import OpenAI
from dotenv import load_dotenv
import logging
import os

# Set up logging
logging.basicConfig(
    filename="../logs/llm_calls_old.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

load_dotenv()


def call_claude(user_prompt, system_prompt="", client=None, model="claude-3-sonnet-20240229", max_tokens=2048):
    if client is None:
        client = anthropic.Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        system=system_prompt
    )
    retval = message.content
    logging.info(f"Called Claude with prompt: {user_prompt}, system prompt: {system_prompt}")
    logging.info(f"Response from Claude: {retval}")
    return message.content


def call_openai(user_prompt, system_prompt="", client=None, model="gpt-4-0125-preview", max_tokens=2048):
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens
    )
    retval = response.choices[0].message.content
    logging.info(f"Called OpenAI with user prompt: {user_prompt}, system prompt: {system_prompt}")
    logging.info(f"Response from OpenAI: {retval}")
    return retval
