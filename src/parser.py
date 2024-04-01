import re


def extract_code_segments(markdown_content):
    """
    Extracts code segments from a given markdown content string.

    Parameters:
    - markdown_content (str): The content of the markdown file as a string.

    Returns:
    - List[str]: A list of extracted code segments.
    """
    # Regular expression pattern for code blocks, ignoring optional language identifiers
    pattern = re.compile(r"```.*?\n(.*?)```", re.DOTALL)

    # Find all matches and return them
    matches = pattern.findall(markdown_content)
    return matches


if __name__ == "__main__":
    # Read the markdown file
    with open("./../inputs/q.md", "r") as file:
        content = file.read()
    print(len(extract_code_segments(content)))
