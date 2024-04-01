from parser import extract_code_segments
from llms import call_openai, call_claude
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def read_file(filename):
    with open(filename, "r", encoding="latin-1") as file:
        return file.read()


def dump_to_file(filename, content):
    """
    Dump in append mode
    :param filename:
    :param content:
    :return:
    """
    with open(filename, "a+") as file:
        file.write(content + "\n\n" + "=" * 80 + "\n\n")


def process_question(question, answering_system_prompt, writing_system_prompt, gpt):
    print("Processing question...")
    try:
        # call the two language models
        gpt_first = call_openai(question, answering_system_prompt, gpt)
        # only extract what is after the ### Answer: tag
        try:
            gpt_first = gpt_first.split("### Answer")[1]
        except:
            pass
        formatted = f"<question>\n\n{question}\n\n</question>\n\n<explanatory_answer>\n\n{gpt_first}\n\n</explanatory_answer>\n\n"
        gpt_writing = call_openai(formatted, writing_system_prompt, gpt)
        # now, write this out to a file, in a code block
        dump_to_file("./../outputs/answers.md", gpt_writing)
    except Exception as e:
        print(e)
        print(question)


def main():
    # read the markdown file and system prompts
    questions = read_file("./../inputs/q.md")
    answering_system_prompt = read_file("./../inputs/gpt_system_prompt.txt")
    writing_system_prompt = read_file("./../inputs/claude_system_prompt.txt")
    # extract the code segments
    segments = extract_code_segments(questions)
    gpt = OpenAI()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_question, question, answering_system_prompt, writing_system_prompt, gpt) for
                   question in segments]

        for future in as_completed(futures):
            try:
                future.result()  # This could be used to get the result or catch exceptions
            except Exception as exc:
                print(f'Generated an exception: {exc}')


if __name__ == "__main__":
    main()
