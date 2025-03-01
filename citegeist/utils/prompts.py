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


def generate_summary_prompt_question_with_page_content(
        question: str, abstract_to_be_considered: str, page_text_to_be_cited: list[str]
) -> str:
    """
    Generates the summary prompt for a given pair of abstracts.
    :param abstract_source_paper: Abstract of source paper
    :param abstract_to_be_cited: Abstract of a related work
    :return: Prompt string
    """
    output = f"""
    Below is a question and the abstract of a paper which may contain relevant information:
    My question:
    "{question}"

    Abstract of the paper that may contain relevant information:
    "{abstract_to_be_considered}"

    Relevant content of {len(page_text_to_be_cited)} pages within the paper:
    """

    for i in range(len(page_text_to_be_cited)):
        text = page_text_to_be_cited[i]
        output += f"""
        Page {i + 1}:
        "{text}"
        """

    output += f"""

    Based on the question and the paper abstract, write a brief few-sentence summary of the abstract in relation to the question. If the abstract does not contain relevant information, please reply with 'No relevant Information'.

    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """

    return output


def generate_summary_prompt_with_page_content(
        abstract_source_paper: str,
        abstract_to_be_cited: str,
        page_text_to_be_cited: list[str],
        sentence_count: int = 8
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
        Page {i + 1}:
        "{text}"
        """

    output += f"""
    Based on the two abstracts and the content from the page, write a brief few-sentence (at most {str(sentence_count)}) summary of the cited paper in relation to my work. Emphasize how the cited paper relates to my research.

    Please exclusively respond with the summary. Do not add any filler text before or after the summary. Also, do not use any type of markdown formatting. I want a pure text output only.
    """
    return output


def generate_question_answer_prompt(question: str, data: list[dict]):
    output = f"""
    I am wondering about a scientific question, and I need a well-written answer that is based on scientific papers. Below I'm providing you with my question and a list of summaries of potentially relevant papers I've identified.

    Here's the question:
    "{question}"

    Here's the list of summaries of the other related works I've found:
    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i + 1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += f"""

    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured answer to the given question. 
    Draw connections between the related papers and my question. 
    If multiple related works have a common point/theme, make sure to group them and refer to them in the same paragraph. 
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes, do not use [x]).
    Group papers with similar topics or implications into the same paragraph. 
    """
    return output


def generate_related_work_prompt(source_abstract: str, data: list[dict], paragraph_count: int = 5,
                                 add_summary: bool = True) -> str:
    """
    Generates the related work prompt for an abstract and a set of summaries & citation strings.
    :param source_abstract: Abstract of source paper
    :param data: List of objects that each contain a paper summary and the respective citation string
    :return: Prompt string
    """
    output = f"""
    I am working on a research paper, and I need a well-written "Related Work" section. Below I'm providing you with the abstract of my paper and a list of summaries of related works I've identified.

    Here's the abstract of my paper:
    "{source_abstract}"

    Here's the list of summaries of the other related works I've found:
    """

    for i in range(len(data)):
        summary = data[i]["summary"]
        citation = data[i]["citation"]
        output += f"""
        Paper {i + 1}:
        Summary: {summary}
        Citation: {citation}
        """

    output += f"""

    Instructions:
    Using all the information given above, your goal is to write a cohesive and well-structured "Related Work" section. 
    Draw connections between the related papers and my research and highlight similarities and differences. 
    If multiple related works have a common point/theme, make sure to group them and refer to them in the same paragraph. 
    Please ensure that your generated section employs all the papers from above.
    When referring to content from specific papers you must also cite the respective paper properly (i.e. cite right after your direct/indirect quotes, do not use [x]).
    Group papers with similar topics or implications into the same paragraph. Limit yourself to at most {str(paragraph_count)} paragraphs, which should not be too short (e.g. avoid 2/3-sentence paragraphs).
    """
    if add_summary:
        output += "Please also make sure to put my work into the overall context of the provided related works in a summarizing paragraph at the end."
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
        Paper {i + 1}:
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


def generate_relevance_evaluation_prompt(source_abstract: str, target_abstract: str) -> str:
    """
    Generates an evaluation prompt to utilize LLM as a judge to determine the relevance with regard to the source abstract
    :param source_abstract: Abstract of source paper
    :param target_abstract: Abstract of target paper
    :return: Prompt String
    """
    prompt = f"""
        You are given two paper abstracts: the first is the source paper abstract, and the second is a related work paper abstract. Your task is to assess the relevance of the related work abstract to the source paper abstract on a scale of 0 to 10, where:

        - 0 means no relevance at all (completely unrelated).
        - 10 means the highest relevance (directly related and closely aligned with the source paper's topic and content).

        Consider factors such as:
        - Topic alignment: Does the related work paper address a similar research problem or area as the source paper?
        - Methodology: Does the related work discuss methods or techniques similar to those in the source paper?
        - Findings or contributions: Are the findings or contributions of the related work closely related to the source paper's content or conclusions?
        - The relationship between the two papers, such as whether the related work builds on, contrasts, or expands the source paper's work.

        Provide a score (0–10) and a brief explanation of your reasoning for the assigned score.

        Source Paper Abstract:
        {source_abstract}

        Related Work Paper Abstract:
        {target_abstract}

        Please provide only the score as your reply. Do not produce any other output, including things like formatting or markdown. Only the score.
    """
    return prompt


def generate_win_rate_evaluation_prompt(source_abstract: str, source_related_work: str, target_related_work: str,
                                        reverse_order: bool = False) -> tuple[str, list[str]]:
    order = []
    if reverse_order:
        order = ['target', 'source']
        tmp = source_related_work
        source_related_work = target_related_work
        target_related_work = tmp
    else:
        order = ['source', 'target']

    return f"""
    Source Abstract:
    {source_abstract}

    Related Works Section A:
    {source_related_work}

    Related Works Section B:
    {target_related_work}

    Objective:
    Evaluate which related works section better complements the source abstract provided.

    Consider factors such as comprehensiveness, clarity of writing, relevance, etc. when making your decision.
    If invalid citations occur, consider the information to be invalid (or even completely false!).

    Exclusively respond with your choice of rating from one of the following options:
        •	Section A
        •	Section B
        •	Tie

    Do not include anything else in your output.
    """, order


def generate_related_work_score_prompt(source_abstract: str, related_work: str) -> str:
    return f"""
    Source Abstract:
    {source_abstract}

    Related Works Section:
    {related_work}

    Objective:
    Evaluate this related works section with regard to the source abstract provided.

    Consider factors such as comprehensiveness, clarity of writing, relevance, etc. when making your decision.
    If invalid citations occur, consider the information to be invalid (or even completely false).

    Exclusively respond with your choice of rating. For this purpose you can assign a score from 0-10 where 0 is worst and 10 is best.

    - **0**: Completely irrelevant, unclear, or inaccurate.  
    *Example*: The section does not address the Source Abstract's topics and contains multiple invalid citations.  

    - **5**: Somewhat relevant but lacks comprehensiveness, clarity or relevance.  
    *Example*: The section references a few relevant works but also includes irrelevant ones and has minor errors.  

    - **10**: Exceptionally relevant, comprehensive, clear, and accurate.  
     *Example*: The section thoroughly addresses all key topics, includes all relevant works, and is clearly written with no factual errors.

    Do not include anything else in your output.
    """