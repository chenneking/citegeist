import json


def load_api_key(file_path: str) -> str:
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["secret_key"]


def generate_summary_prompt(abstract_source_paper: str, abstract_to_be_cited: str):
    return f'''
    Below are two abstracts:
    My abstract:
    "{abstract_source_paper}"
    Abstract of the paper I want to cite:
    "{abstract_to_be_cited}"
    Based on the two abstracts, write a brief few-sentence (at most 5) summary of the cited paper in relation to my work. Emphasize how the cited paper relates to my research.
    
    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    '''


def generate_related_work_prompt(source_abstract: str, data: list[object]) -> str:
    output = f'''
    I am working on a research paper, and I need a well-written "Related Work" section. Below, I provide:
    The abstract of my paper:
    "{source_abstract}"
    '''

    for i in range(len(data)):
        summary = data[i]['summary']
        citation = data[i]['citation']
        output += f'''
        
        Paper {i+1}:
        Summary: {summary}
        Citation: {citation}
        '''

    output += '''
    
    Instructions:
    Using the above information:
    Write a cohesive and well-structured "Related Work" section that integrates the provided summaries and citations.
    Make meaningful connections between the related papers and my research, highlighting similarities, differences, and how the related work contextualizes my study.
    Ensure the text flows logically, grouped into thematic paragraphs as needed.
    Use the provided citations where relevant to indicate references to the papers.
    '''
    # summarize multiple related papers/citations into one paragraph
    # don't be as repetitive

    return output