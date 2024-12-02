import json


def load_api_key(file_path: str) -> str:
    """
    Retrieves the API key from the given file.
    :param file_path: file path to key file
    :return: API key
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["secret_key"]


def generate_summary_prompt(
    abstract_source_paper: str, abstract_to_be_cited: str
) -> str:
    """
    Generates the summary prompt for a given pair of abstracts.
    :param abstract_source_paper: Abstract of source paper
    :param abstract_to_be_cited: Abstract of a related work
    :return: Prompt string
    """
    return f"""
    Below are two abstracts:
    My abstract:
    "{abstract_source_paper}"
    Abstract of the paper I want to cite:
    "{abstract_to_be_cited}"
    Based on the two abstracts, write a brief few-sentence (at most 5) summary of the cited paper in relation to my work. Emphasize how the cited paper relates to my research.
    
    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """


def generate_summary_prompt_with_page_content(
    abstract_source_paper: str,
    abstract_to_be_cited: str,
    page_text_to_be_cited: list[str],
) -> str:
    """
    Generates the summary prompt for a given pair of abstracts and a list of relevant pages.
    :param abstract_source_paper: Abstract of source paper
    :param abstract_to_be_cited: Abstract of a related work
    :param page_text_to_be_cited: List of page text(s) of the related work
    :return: Prompt string
    """
    output = f"""
    Below are two abstracts and some content from a page of a paper:
    My abstract:
    "{abstract_source_paper}"
    
    Abstract of the paper I want to cite:
    "{abstract_to_be_cited}"
    
    Relevant content of {len(page_text_to_be_cited)} pages within the paper I want to cite:
    """

    for i in range(len(page_text_to_be_cited)):
        text = page_text_to_be_cited[i]
        output += f"""
        Page {i+1}:
        "{text}"
        """

    output += f"""
    Based on the two abstracts and the content from the page, write a brief few-sentence (at most 8) summary of the cited paper in relation to my work. Emphasize how the cited paper relates to my research.

    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """
    return output


def generate_related_work_prompt(source_abstract: str, data: list[object]) -> str:
    """
    Generates the related work prompt for an abstract and a set of summaries & citation strings.
    :param source_abstract: Abstract of source paper
    :param data: List of objects that each contain a paper summary and the respective citation string
    :return: Prompt string
    """
    output = f"""
    I am working on a research paper, and I need a well-written "Related Work" section. Below I'm providing you with the abstract of the paper I'm writing and a list of summaries of related works I've identified.
    
    Here's the abstract of my paper:
    "{source_abstract}"
    
    Here's the list of summaries of the other related works I've found:
    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i+1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += """
    
    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured "Related Work" section. 
    Draw connections between the related papers and my research and highlight similarities and differences. 
    Please also make sure to put my work into the overall context of the provided related works in a summarizing paragraph at the end. 
    If multiple related works have a common point/theme, make sure to group them and refer to them in the same paragraph. 
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes).
    Insert paragraphs where appropriate which are concise, but not too short (e.g. avoid 2-sentence paragraphs).
    """
    return output

def generate_related_work_analysis_prompt(source_abstract: str, data: list[object]) -> str:
    """
    Generates the related work analysis prompt for an abstract and a set of summaries & citation strings.
    :param source_abstract: Abstract of source paper
    :param data: List of objects that each contain a paper summary and the respective citation string
    :return: Prompt string
    """
    output = f"""
    I am working on a research paper, and I would like to get a sense of related work for a specific section of my paper. Below I'm providing you with the section whose background I am interested in and a list of summaries of related works I've identified.
    
    Here's the section of my paper:
    "{source_abstract}"
    
    Here's the list of summaries of the other related works I've found:
    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i+1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += """
    
    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured analysis of the related work. 
    Draw connections between the related papers and my research section and highlight similarities and differences. 
    Please also make sure to put my section into the overall context of the provided related works in a summarizing paragraph at the end. 
    If multiple related works have a common points or themes, make sure to group them and refer to them in the same paragraph. 
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes).
    """
    return output


def read_json_file(file_path: str) -> dict | list | None:
    """
    Reads a JSON file from the given path.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict or list: Parsed JSON data (dictionary or list depending on the JSON structure).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    return None
