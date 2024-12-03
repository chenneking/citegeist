import pandas as pd
from utils.azure_client import AzureClient
from utils.helpers import generate_win_rate_evaluation_prompt, load_api_key
import os
from dotenv import load_dotenv

load_dotenv()

papers_df = pd.read_csv('sample_data/papers.csv')
output_df = pd.read_csv('sample_data/output.csv')

prompting_client = AzureClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        deployment_id=os.getenv("AZURE_PROMPTING_MODEL"),
        api_key=load_api_key(os.getenv("KEY_LOCATION")),
    )

for i, row in papers_df.iterrows():
    abstract = row['Abstract']
    source_related_works = row['Related Works']
    gpt_related_works = row['Related Works (GPT-4o-min)']

    output_row = output_df.loc[i]
    our_related_works = output_row['related_works']

    win_rate_prompt, order = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=source_related_works,
        target_related_work=our_related_works
    )

    win_rate_response: str = prompting_client.get_completions(
        win_rate_prompt,
        os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response)
    # Modify the results to account for reordering
    selected_winner_paper_source_ours: str = ''
    if win_rate_response == 'Section A':
        if order[0] == 'source':
            selected_winner_paper_source_ours = order[0]
        elif order[0] == 'target':
            selected_winner_paper_source_ours = 'our_approach'
    elif win_rate_response == 'Section B':
        if order[1] == 'source':
            selected_winner_paper_source_ours = order[1]
        elif order[1] == 'target':
            selected_winner_paper_source_ours = 'our_approach'
    elif win_rate_response == 'Tie':
        print(f'There was a tie.')
        selected_winner_paper_source_ours = 'tie'

    win_rate_prompt2, order2 = generate_win_rate_evaluation_prompt(
        source_abstract=abstract,
        source_related_work=our_related_works,
        target_related_work=gpt_related_works
    )

    win_rate_response2: str = prompting_client.get_completions(
        win_rate_prompt2,
        os.getenv("AZURE_PROMPTING_MODEL_VERSION")
    )

    print(win_rate_response2)
    # Modify the results to account for reordering
    selected_winner_paper_ours_gpt: str = ''
    if win_rate_response2 == 'Section A':
        if order2[0] == 'source':
            selected_winner_paper_ours_gpt = 'our_approach'
        elif order2[0] == 'target':
            selected_winner_paper_ours_gpt = 'gpt'
    elif win_rate_response2 == 'Section B':
        if order2[1] == 'source':
            selected_winner_paper_ours_gpt = 'our_approach'
        elif order2[1] == 'target':
            selected_winner_paper_ours_gpt = 'gpt'
    elif win_rate_response2 == 'Tie':
        print(f'There was a tie.')
        selected_winner_paper_ours_gpt = 'tie'


    print(f'Paper {i}: {selected_winner_paper_source_ours}, {selected_winner_paper_ours_gpt}')
    print('------------')

