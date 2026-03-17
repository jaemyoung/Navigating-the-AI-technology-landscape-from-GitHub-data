# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:46:37 2022
Modified by AI assistant on 2025-05-20 to incorporate new filters and token management.
@author: user
"""
import os
import time
import datetime
import pickle
import random
import re # For sanitizing filenames
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import pandas as pd
from github import Github, RateLimitExceededException, GithubException

# --- Configuration ---
ACCESS_TOKENS = [
    "YOUR_GITHUB_TOKEN_1",
    "YOUR_GITHUB_TOKEN_2",
    # Add more tokens as needed
]
CURRENT_TOKEN_INDEX = 0
g = None # Global Github client instance

# --- Search Parameters ---
# Keywords inspired by the second script
PLAIN_KEYWORDS = [
    '"artificial intelligence" OR "computer vision" OR "deep learning" OR "machine learning" OR "image recognition"',
    '"natural language processing" OR "nlp" OR "language model" OR "speech recognition"',
]
AI_ADJECTIVES = [
    "agentic", "causal", "explainable", "generative", "multi-modal",
    "quantum", "reliable", "responsible", "robotics", "symbolic"
]
COMBINED_AI_KEYWORDS = [f'"{adj}" AND "AI"' for adj in AI_ADJECTIVES] # Phrase search for "adj AI"

KEYWORD_QUERIES_TO_PROCESS = COMBINED_AI_KEYWORDS

# Fixed Filters
SEARCH_IN_FILTER = "in:name,description,readme"
LANGUAGE_FILTER = "language:python"
STARS_FILTER = "stars:>=10"
PUSHED_FILTER = "pushed:2020-01-01..2024-12-31"

# Date range for 'created:' filter
CREATED_START_DATE = "2021-01-01"
CREATED_END_DATE = "2024-12-31" # Current date: 2025-05-20
MONTHS_PER_PERIOD = 1 # 1개월 단위로 분할

# Rate Limiting / Pausing
API_CALLS_BEFORE_REST = 280 # 한 토큰 당 API 호출 후 휴식 임계값 (PyGithub search는 1회당 약 1~2 API call 소모)
PRIMARY_REST_DURATION_SECONDS = 120 # 기본 휴식 시간
TOKEN_EXHAUSTED_REST_DURATION_SECONDS = 300 # 토큰 모두 소진 시 휴식 시간
INTER_REPO_SLEEP_MIN = 0.1 # 각 레포지토리 처리 후 최소 대기 시간
INTER_REPO_SLEEP_MAX = 0.5 # 각 레포지토리 처리 후 최대 대기 시간

# Output Files
os.makedirs('crawled_data_2021_2024', exist_ok=True) # 새 폴더에 저장

def sanitize_filename(text):
    """Sanitizes a string to be part of a filename."""
    sanitized = re.sub(r'[^a-zA-Z0-9_\-\s]', '', text)
    sanitized = re.sub(r'\s+', '_', sanitized).strip('_')
    return sanitized[:50]

def initialize_github_client(token_index_to_try):
    global g, CURRENT_TOKEN_INDEX
    if not ACCESS_TOKENS:
        print("CRITICAL Error: No access tokens provided.")
        return None
    if token_index_to_try >= len(ACCESS_TOKENS):
        print(f"Warning: Token index {token_index_to_try} out of bounds.")
        return None

    token = ACCESS_TOKENS[token_index_to_try]
    print(f"\nAttempting to initialize GitHub client with token index {token_index_to_try} (...{token[-4:]})")
    try:
        client = Github(token, timeout=30, per_page=100)
        # Test call to check token validity and rate limit
        rate_limit = client.get_rate_limit()
        print(f"GitHub client initialized. Rate limit: {rate_limit.core.remaining}/{rate_limit.core.limit}. Reset: {rate_limit.core.reset}")
        if rate_limit.core.remaining < 10: # 너무 적으면 다음 토큰으로
             print(f"Token index {token_index_to_try} has too few remaining calls ({rate_limit.core.remaining}).")
             return None
        g = client
        CURRENT_TOKEN_INDEX = token_index_to_try
        return g
    except RateLimitExceededException:
        print(f"Token index {token_index_to_try} is rate-limited upon initialization.")
        return None
    except GithubException as e:
        if e.status == 401:
            print(f"Token index {token_index_to_try} is invalid (401 Unauthorized).")
        else:
            print(f"GitHubException for token index {token_index_to_try}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error initializing client with token index {token_index_to_try}: {e}")
        return None

def get_next_github_client():
    global g, CURRENT_TOKEN_INDEX
    if not ACCESS_TOKENS: return None

    initial_idx = CURRENT_TOKEN_INDEX
    for i in range(len(ACCESS_TOKENS)):
        next_idx = (initial_idx + 1 + i) % len(ACCESS_TOKENS) # Start from next token
        client = initialize_github_client(next_idx)
        if client:
            return client
    
    print(f"\nAll {len(ACCESS_TOKENS)} tokens seem exhausted or invalid. Waiting for {TOKEN_EXHAUSTED_REST_DURATION_SECONDS} seconds.")
    time.sleep(TOKEN_EXHAUSTED_REST_DURATION_SECONDS)
    # Retry from the first token after long sleep
    CURRENT_TOKEN_INDEX = -1 # So that ((-1)+1+0) % len = 0 for the next attempt
    client = initialize_github_client(0) # Try initializing with the first token again
    return client


def make_search_periods(start_str, end_str, months_interval):
    """입력된 기간을 원하는 개월로 나눠 GitHub 검색용 'start..end' 문자열 리스트 생성"""
    try:
        start_dt = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
        end_dt = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        return []

    if start_dt >= end_dt:
        print("Error: Start date must be before end date.")
        return []

    periods = []
    current_period_start = start_dt
    while current_period_start <= end_dt: # Use <= to include the last month if it aligns
        period_end_exclusive = current_period_start + relativedelta(months=months_interval)
        # GitHub API '..' range is inclusive for the end date.
        # So, use period_end_exclusive - 1 day, or overall_end_date if sooner.
        current_period_end = period_end_exclusive - datetime.timedelta(days=1)

        if current_period_end > end_dt:
            current_period_end = end_dt
        
        periods.append(f"{current_period_start.isoformat()}..{current_period_end.isoformat()}")
        
        if current_period_end >= end_dt: # Ensure we don't go past the overall end_dt
            break
        
        current_period_start = period_end_exclusive # Next period starts where the previous one ended (exclusive)
        if months_interval <= 0:
            print("Error: months_interval must be positive.")
            break
            
    print(f"Generated {len(periods)} 'created:' date periods from {start_str} to {end_str}.")
    return periods


def crawl_repositories_for_keyword(keyword_query, created_periods):
    global g, CURRENT_TOKEN_INDEX

    repo_list_for_keyword = []
    api_calls_on_current_token = 0
    total_repos_processed_for_kw = 0

    if not g:
        print("Initial GitHub client is None. Attempting to get one.")
        g = get_next_github_client()
        if not g:
            print("CRITICAL: Could not obtain GitHub client. Skipping keyword.")
            return []

    print(f"\n=== Processing Keyword: {keyword_query} ===")
    
    for period_idx, created_period_str in enumerate(tqdm(created_periods, desc=f"Periods for '{sanitize_filename(keyword_query)}'")):
        base_query_parts = [
            keyword_query,
            SEARCH_IN_FILTER,
            LANGUAGE_FILTER,
            STARS_FILTER,
            PUSHED_FILTER,
            f"created:{created_period_str}"
        ]
        search_query_full = " ".join(filter(None, base_query_parts))
        
        print(f"\nPeriod {period_idx+1}/{len(created_periods)}: {created_period_str}")
        print(f"Search query: {search_query_full}")

        retries_for_period = 0
        max_retries_for_period = 2 # 한 기간에 대해 최대 재시도 횟수

        while retries_for_period <= max_retries_for_period:
            try:
                if api_calls_on_current_token >= API_CALLS_BEFORE_REST:
                    print(f"API call limit ({API_CALLS_BEFORE_REST}) for current token reached. Resting for {PRIMARY_REST_DURATION_SECONDS}s.")
                    time.sleep(PRIMARY_REST_DURATION_SECONDS)
                    api_calls_on_current_token = 0
                    # Potentially re-check token or switch if needed, but get_next_github_client handles this on failure.

                repositories_page = g.search_repositories(query=search_query_full, sort="stars", order="desc")
                api_calls_on_current_token +=1 # search_repositories is usually 1 API call unless paginating heavily internally

                count_for_period = 0
                if repositories_page.totalCount > 0:
                    print(f"Found ~{repositories_page.totalCount} repositories for this period. Fetching up to 1000.")
                    for repo_idx, repo in enumerate(repositories_page):
                        if repo_idx >= 1000: # GitHub API limit
                            print("Reached 1000 repositories for this specific query, stopping for this period.")
                            break
                        repo_list_for_keyword.append(repo)
                        count_for_period += 1
                        total_repos_processed_for_kw +=1
                        # print(f"  Collected {repo.full_name} ({total_repos_processed_for_kw} for keyword)")
                        time.sleep(random.uniform(INTER_REPO_SLEEP_MIN, INTER_REPO_SLEEP_MAX))
                else:
                    print("No repositories found for this period and query.")
                
                print(f"Collected {count_for_period} repositories in period {created_period_str}.")
                break # Success for this period, move to next period

            except RateLimitExceededException:
                print(f"RateLimitExceeded for token {CURRENT_TOKEN_INDEX} (call {api_calls_on_current_token}). Switching token.")
                api_calls_on_current_token = 0 # Reset for new token
                g = get_next_github_client()
                if not g:
                    print("CRITICAL: No usable tokens left after rate limit. Aborting for this keyword.")
                    return repo_list_for_keyword # Return what's collected so far for this keyword
                # Continue to retry current period with new token
                
            except GithubException as e:
                print(f"GithubException occurred: {e} (Status: {e.status})")
                if e.status == 401: # Bad credentials
                    print(f"Token {CURRENT_TOKEN_INDEX} is invalid. Switching token.")
                    api_calls_on_current_token = 0
                    g = get_next_github_client()
                    if not g:
                        print("CRITICAL: No usable tokens. Aborting for this keyword.")
                        return repo_list_for_keyword
                elif e.status == 422: # Unprocessable Entity (often bad query syntax or too complex)
                     print(f"Query validation failed (422): {search_query_full}. Skipping this period.")
                     break # Skip this period
                else: # Other GitHub errors
                    print("Pausing due to other GitHub error and retrying period...")
                    time.sleep(PRIMARY_REST_DURATION_SECONDS)
                    # Retry with the same token unless it's deemed problematic by get_next_github_client
                
            except Exception as e: # Catch other errors like network issues
                print(f"An unexpected error occurred: {e}. Pausing and retrying period.")
                time.sleep(PRIMARY_REST_DURATION_SECONDS * 2) # Longer pause for unexpected network issues
            
            retries_for_period += 1
            if retries_for_period > max_retries_for_period:
                 print(f"Max retries reached for period {created_period_str}. Skipping this period.")

        # Save intermediate list for current keyword after processing all its periods, or every N repos
        if len(repo_list_for_keyword) > 0 and (len(repo_list_for_keyword) % 2000 == 0 or period_idx == len(created_periods) -1 ):
            filename_sanitized = sanitize_filename(keyword_query)
            filepath = f'crawled_data_2021_2024/repo_list_intermediate_{filename_sanitized}_{len(repo_list_for_keyword)}.pkl'
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(repo_list_for_keyword, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Intermediate repo list for '{keyword_query}' saved to {filepath}")
            except Exception as e_save:
                print(f"Error saving intermediate list: {e_save}")

    return repo_list_for_keyword


def extract_metadata_from_repos(repo_object_list, keyword_query_source="N/A"):
    """레포지토리 객체 리스트에서 메타데이터를 추출하여 DataFrame으로 반환"""
    crawled_data = []
    print(f"\nExtracting metadata for {len(repo_object_list)} repositories from keyword '{keyword_query_source}'...")
    
    for idx, repo in enumerate(tqdm(repo_object_list, desc="Extracting Metadata")):
        try:
            # Basic metadata, similar to original script's first crawling_data
            row_data = {
                "total_index": idx,
                "full_name": repo.full_name,
                "create_date": repo.created_at.isoformat() if repo.created_at else None,
                "update_date": repo.updated_at.isoformat() if repo.updated_at else None,
                "pushed_date": repo.pushed_at.isoformat() if repo.pushed_at else None, # Added
                "language": repo.language,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count, # Added
                "open_issues": repo.open_issues_count, # Added
                "description": repo.description,
                "topics": repo.get_topics() if hasattr(repo, 'get_topics') else [], # Added
                "html_url": repo.html_url, # Added
                "is_fork": repo.fork, # Added
                "keyword_source": keyword_query_source # Added to know which query found this
            }
            crawled_data.append(row_data)
        except Exception as e:
            print(f"Error extracting metadata for {getattr(repo, 'full_name', 'UNKNOWN_REPO')}: {e}")
            # Add a partial entry or skip
            crawled_data.append({
                "total_index": idx,
                "full_name": getattr(repo, 'full_name', f'ERROR_REPO_{idx}'),
                "description": f"Error: {e}",
                "keyword_source": keyword_query_source
            })
            
    if not crawled_data:
        return pd.DataFrame()

    df = pd.DataFrame(crawled_data)
    # Reorder columns for better readability
    cols_ordered = [
        "total_index", "keyword_source", "full_name", "html_url", "create_date", "update_date", "pushed_date",
        "language", "stars", "forks", "watchers", "open_issues", "is_fork",
        "topics", "description"
    ]
    # Add any missing columns from the ideal order (e.g. if all extractions failed for a field)
    for col in cols_ordered:
        if col not in df.columns:
            df[col] = None
            
    return df[cols_ordered]


if __name__ == '__main__':
    print("--- GitHub Repository Crawler (Modified) ---")
    start_time_overall = time.time()

    # Initialize first client
    g = initialize_github_client(CURRENT_TOKEN_INDEX)
    if not g:
        g = get_next_github_client() # Try to get any working client
    
    if not g:
        print("CRITICAL: No working GitHub tokens available at startup. Exiting.")
        exit()

    # Generate 'created:' date periods once
    created_date_periods = make_search_periods(CREATED_START_DATE, CREATED_END_DATE, MONTHS_PER_PERIOD)
    if not created_date_periods:
        print("CRITICAL: Could not generate 'created:' date periods. Exiting.")
        exit()

    all_collected_metadata_df = pd.DataFrame()

    for kw_idx, keyword_q in enumerate(KEYWORD_QUERIES_TO_PROCESS):
        print(f"\n\n{'='*60}")
        print(f"STARTING KEYWORD QUERY {kw_idx+1}/{len(KEYWORD_QUERIES_TO_PROCESS)}: {keyword_q}")
        print(f"{'='*60}")
        
        start_time_keyword = time.time()
        
        # Crawl repositories for the current keyword and all created_periods
        repo_objects = crawl_repositories_for_keyword(keyword_q, created_date_periods)
        
        if repo_objects:
            print(f"\nFound {len(repo_objects)} raw repository objects for keyword: '{keyword_q}'.")
            
            # Deduplicate repository objects (based on full_name as a simple proxy for ID)
            unique_repo_objects_dict = {repo.full_name: repo for repo in repo_objects}
            unique_repo_objects_list = list(unique_repo_objects_dict.values())
            print(f"After deduplication: {len(unique_repo_objects_list)} unique repositories for keyword: '{keyword_q}'.")

            if unique_repo_objects_list:
                # Extract metadata
                metadata_df_for_keyword = extract_metadata_from_repos(unique_repo_objects_list, keyword_q)
                
                if not metadata_df_for_keyword.empty:
                    # Save data for this keyword
                    sanitized_kw_filename = sanitize_filename(keyword_q)
                    
                    # Save repo objects list (pickle)
                    repo_list_filepath = f'crawled_data_2021_2024/FINAL_repo_objects_{sanitized_kw_filename}_{len(unique_repo_objects_list)}.pkl'
                    with open(repo_list_filepath, 'wb') as f:
                        pickle.dump(unique_repo_objects_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Final unique repo object list for '{keyword_q}' saved to {repo_list_filepath}")

                    # Save metadata (CSV and pickle)
                    metadata_csv_filepath = f'crawled_data_2021_2024/FINAL_metadata_{sanitized_kw_filename}_{len(metadata_df_for_keyword)}.csv'
                    metadata_pkl_filepath = f'crawled_data_2021_2024/FINAL_metadata_{sanitized_kw_filename}_{len(metadata_df_for_keyword)}.pkl'
                    
                    metadata_df_for_keyword.to_csv(metadata_csv_filepath, index=False, encoding='utf-8-sig')
                    print(f"Metadata for '{keyword_q}' saved to {metadata_csv_filepath}")
                    metadata_df_for_keyword.to_pickle(metadata_pkl_filepath)
                    print(f"Metadata for '{keyword_q}' saved to {metadata_pkl_filepath}")
                    
                    # Append to a combined DataFrame (optional, can be large)
                    # all_collected_metadata_df = pd.concat([all_collected_metadata_df, metadata_df_for_keyword], ignore_index=True)

                else:
                    print(f"No metadata extracted for keyword '{keyword_q}'.")
            else:
                print(f"No unique repositories to process metadata for keyword '{keyword_q}'.")
        else:
            print(f"No repositories collected for keyword '{keyword_q}'.")

        end_time_keyword = time.time()
        print(f"Time taken for keyword '{keyword_q}': {end_time_keyword - start_time_keyword:.2f} seconds.")
        
        if kw_idx < len(KEYWORD_QUERIES_TO_PROCESS) - 1:
            inter_keyword_pause = random.uniform(10, 20)
            print(f"Pausing for {inter_keyword_pause:.2f}s before next keyword.")
            time.sleep(inter_keyword_pause)

    # if not all_collected_metadata_df.empty:
    #     all_collected_metadata_df.to_csv('crawled_data_2021_2024/ALL_KEYWORDS_COMBINED_metadata.csv', index=False, encoding='utf-8-sig')
    #     all_collected_metadata_df.to_pickle('crawled_data_2021_2024/ALL_KEYWORDS_COMBINED_metadata.pkl')
    #     print("\nCombined metadata for all keywords saved.")

    end_time_overall = time.time()
    print(f"\n\n--- Crawler Finished ---")
    print(f"Total execution time: {end_time_overall - start_time_overall:.2f} seconds.")