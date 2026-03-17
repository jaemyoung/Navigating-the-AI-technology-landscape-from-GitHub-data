# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:16:42 2022
Modified by AI to load multiple pkl files and handle multiple GitHub tokens.

@author: user
"""
#%% package
import os
import pickle
import re
import time # time.sleep을 위해 추가
import numpy as np
from tqdm import tqdm
import pandas as pd
import requests
from requests.exceptions import HTTPError
from collections import Counter
from bs4 import BeautifulSoup
import networkx as nx # 명시적으로 사용되지 않더라도, 원래 코드에 있었으므로 유지
from github import Github, BadCredentialsException, RateLimitExceededException, GithubException, UnknownObjectException # UnknownObjectException 추가됨
import matplotlib.pyplot as plt
import traceback # 상세한 오류 출력을 위해 추가

#%% Github token - Multi-token handling


github_tokens =[
    "YOUR_GITHUB_TOKEN_1",
    "YOUR_GITHUB_TOKEN_2",
    # Add more tokens as needed
]

g = None # Github API 객체를 저장할 변수
current_token_index = 0 # 현재 사용 중인 토큰 인덱스

print("GitHub API 연결을 시도합니다...")
if not github_tokens or all(not token or token.startswith("YOUR_") or token.startswith("ghp_xxxxxxxxxx") for token in github_tokens):
    print("치명적 오류: 유효한 GitHub 토큰이 `github_tokens` 리스트에 설정되지 않았습니다. 스크립트를 종료합니다.")
    exit()

for i, token_str in enumerate(github_tokens):
    if not token_str or token_str.startswith("YOUR_") or token_str.startswith("ghp_xxxxxxxxxx"):
        print(f"토큰 {i+1}이(가) 유효하게 설정되지 않았거나 플레이스홀더입니다. 건너뜁니다.")
        continue

    print(f"토큰 {i+1} ({token_str[:8]}...)을(를) 사용하여 GitHub API 연결 시도 중...")
    try:
        g_candidate = Github(token_str, timeout=30) # API 타임아웃 설정
        user = g_candidate.get_user() # API 연결 테스트
        print(f"성공: GitHub API에 '{user.login}' 사용자로 연결되었습니다 (토큰 {i+1} 사용).")
        g = g_candidate # 성공한 객체 할당
        current_token_index = i # 현재 사용된 토큰 인덱스 저장
        break # 성공했으므로 루프 종료
    except BadCredentialsException:
        print(f"오류: 토큰 {i+1} ({token_str[:8]}...)이(가) 유효하지 않습니다 (BadCredentialsException).")
    except RateLimitExceededException:
        print(f"오류: 토큰 {i+1} ({token_str[:8]}...)의 API 요청 한도를 초과했습니다 (RateLimitExceededException).")
    except requests.exceptions.Timeout:
        print(f"오류: 토큰 {i+1} ({token_str[:8]}...) 사용 중 GitHub API 연결 타임아웃 발생.")
    except GithubException as e: # PyGithub 관련 다른 예외
        print(f"오류: 토큰 {i+1} ({token_str[:8]}...) 사용 중 GitHub API 오류 발생: {e}.")
    except requests.exceptions.ConnectionError as e: # 네트워크 연결 오류 포함
        print(f"오류: 토큰 {i+1} ({token_str[:8]}...) 사용 중 네트워크 연결 오류 발생: {e}.")
    except Exception as e: # 그 외 모든 예외
        print(f"오류: 토큰 {i+1} ({token_str[:8]}...) 사용 중 예상치 못한 오류 발생: {e} (타입: {type(e).__name__}).")
    
    if i < len(github_tokens) - 1:
        print("다음 토큰을 시도합니다...")
    time.sleep(1) # 다음 시도 전 잠시 대기

if g is None:
    print("치명적 오류: 사용 가능한 GitHub 토큰이 없습니다. 모든 토큰으로 API 연결에 실패했습니다. 스크립트를 종료합니다.")
    exit()

#%% Load data - Modified to robustly load all pkl files from a folder
source_folder_path = 'crawled_data_2013_2016/' # 데이터 파일이 있는 폴더 경로
all_loaded_objects = [] # 로드된 모든 객체를 임시 저장
pickle_files_to_load_paths = [] # 로드할 파일 경로 저장 (디버깅용)

print(f"\n'{source_folder_path}' 폴더에서 .pkl 파일을 검색합니다...")
if not os.path.isdir(source_folder_path):
    print(f"오류: '{source_folder_path}' 폴더를 찾을 수 없습니다. 스크립트를 종료합니다.")
    exit()

for filename in os.listdir(source_folder_path):
    if filename.endswith(".pkl"):
        pickle_files_to_load_paths.append(os.path.join(source_folder_path, filename))

if not pickle_files_to_load_paths:
    print(f"'{source_folder_path}' 폴더에 불러올 .pkl 파일이 없습니다. 스크립트를 종료합니다.")
    exit()
else:
    print(f"다음 .pkl 파일들을 순서대로 불러옵니다: {pickle_files_to_load_paths}")

for file_idx, file_path in enumerate(pickle_files_to_load_paths):
    print(f"\n파일 로드 시도 ({file_idx+1}/{len(pickle_files_to_load_paths)}): '{file_path}'")
    try:
        loaded_object = pd.read_pickle(file_path)
        all_loaded_objects.append(loaded_object)
        print(f"'{file_path}' 파일에서 객체 (타입: {type(loaded_object)})를 로드했습니다.")
    except ModuleNotFoundError as mnfe:
        print(f"오류: '{file_path}' 로드 중 모듈을 찾을 수 없음: {mnfe}. 이 파일은 건너뜁니다.")
        all_loaded_objects.append(None)
    except pickle.UnpicklingError as upe:
        print(f"오류: '{file_path}' unpickling 중 오류 (파일 손상 또는 호환되지 않는 형식): {upe}. 이 파일은 건너뜁니다.")
        all_loaded_objects.append(None)
    except EOFError as eofe: # 파일이 비어있거나 너무 작을 때 발생 가능
        print(f"오류: '{file_path}' 로드 중 파일 끝 예상치 않게 도달: {eofe}. 이 파일은 건너뜁니다.")
        all_loaded_objects.append(None)
    except Exception as e:
        print(f"오류: '{file_path}' 로드 중 예상치 못한 오류: {e} (타입: {type(e).__name__}). 이 파일은 건너뜁니다.")
        all_loaded_objects.append(None)

valid_dataframes_for_concat = []
print("\n로드된 객체들을 DataFrame으로 변환 및 검증합니다...")
for i, loaded_obj in enumerate(all_loaded_objects):
    file_path_debug = pickle_files_to_load_paths[i]
    if loaded_obj is None:
        print(f"객체 {i+1} (from '{file_path_debug}')는 로드에 실패하여 건너뛰었습니다.")
        continue

    if isinstance(loaded_obj, pd.DataFrame):
        if not loaded_obj.empty:
            valid_dataframes_for_concat.append(loaded_obj)
            print(f"객체 {i+1} (from '{file_path_debug}')는 유효한 DataFrame입니다. (Rows: {len(loaded_obj)}, Columns: {len(loaded_obj.columns)})")
        else:
            print(f"객체 {i+1} (from '{file_path_debug}')는 비어있는 DataFrame입니다. concat에서 제외합니다.")
    elif isinstance(loaded_obj, list):
        if loaded_obj and all(isinstance(item, dict) for item in loaded_obj): # List[dict] 형태인지 확인
            try:
                df_converted = pd.DataFrame(loaded_obj)
                if not df_converted.empty:
                    valid_dataframes_for_concat.append(df_converted)
                    print(f"객체 {i+1} (from '{file_path_debug}')를 List[dict]에서 DataFrame으로 변환 성공. (Rows: {len(df_converted)}, Columns: {len(df_converted.columns)})")
                else:
                    print(f"객체 {i+1} (from '{file_path_debug}') List[dict]에서 변환된 DataFrame이 비어있어 제외합니다.")
            except Exception as conversion_e:
                print(f"오류: 객체 {i+1} (from '{file_path_debug}') List[dict]를 DataFrame으로 변환 중 오류: {conversion_e}. 건너뜁니다.")
        elif not loaded_obj: # 비어있는 리스트
             print(f"객체 {i+1} (from '{file_path_debug}')는 비어있는 List입니다. 건너뜁니다.")
        else:
            print(f"경고: 객체 {i+1} (from '{file_path_debug}')는 DataFrame으로 변환할 수 없는 List 타입입니다 (내부 요소가 dict가 아님). 건너뜁니다.")
    else:
        print(f"경고: 객체 {i+1} (from '{file_path_debug}')는 DataFrame이 아닌 {type(loaded_obj)} 타입이며 변환 로직이 없습니다. 건너뜁니다.")

if not valid_dataframes_for_concat:
    print("concat할 유효한 데이터프레임이 없습니다. 스크립트를 종료합니다.")
    exit()

print(f"\n총 {len(valid_dataframes_for_concat)}개의 유효한 데이터프레임을 합치는 중...")
try:
    data = pd.concat(valid_dataframes_for_concat, ignore_index=True)
    print("모든 유효한 데이터프레임을 성공적으로 합쳤습니다.")
    print(f"합쳐진 데이터프레임의 크기: {data.shape}")
    if not data.empty:
        print("합쳐진 데이터프레임의 처음 5개 행:")
        print(data.head())
        print("\n합쳐진 데이터프레임의 컬럼 정보:")
        data.info()
    else:
        print("합쳐진 데이터프레임이 비어있습니다. 이후 처리에 문제가 발생할 수 있습니다.")
except Exception as concat_e:
    print(f"데이터프레임 concat 중 치명적 오류 발생: {concat_e}")
    print("오류 상세 정보:")
    traceback.print_exc()
    print("\nconcat 대상 DataFrame들의 컬럼 정보를 출력합니다 (오류 원인 파악용):")
    for df_idx, df_check in enumerate(valid_dataframes_for_concat):
        # valid_dataframes_for_concat 리스트 내의 객체 인덱스와 원래 파일 경로 인덱스가 다를 수 있으므로 주의
        # 여기서는 단순 인덱스만 사용
        try:
            print(f"  DataFrame {df_idx}: shape {df_check.shape}, columns: {df_check.columns.tolist()}")
        except Exception as df_info_e:
            print(f"  DataFrame {df_idx} 정보 출력 중 오류: {df_info_e}")
    print("스크립트를 종료합니다.")
    exit()
# --- 데이터 로드 및 병합 로직 끝 ---

#%% Function Definitions
def add_setupfile(repo_df_input):
    global current_token_index, g, github_tokens # 전역 변수 사용 명시
    repo_df = repo_df_input.copy()
    repo_df["setupfile"] = None
    tiredness = 0
    successful_fetches = 0

    if not github_tokens or all(not token or token.startswith("YOUR_") for token in github_tokens):
        print("오류: add_setupfile - 유효한 GitHub 토큰이 없습니다.")
        return repo_df

    for idx, repo_full_name in tqdm(repo_df["full_name"].items(), total=len(repo_df), desc="Fetching setup.py"):
        api_object_for_repo = None
        initial_token_idx_for_repo = current_token_index # 현재 저장소 시도 시작 토큰 인덱스
        
        for attempt in range(len(github_tokens)): # 모든 토큰을 순환하며 시도
            token_to_try_idx = (initial_token_idx_for_repo + attempt) % len(github_tokens)
            token_str_to_try = github_tokens[token_to_try_idx]

            if not token_str_to_try or token_str_to_try.startswith("YOUR_"): # 플레이스홀더 또는 빈 토큰 건너뛰기
                if len(github_tokens) == 1: # 토큰이 하나뿐인데 유효하지 않으면 더 이상 시도할 수 없음
                    print(f"  오류: 저장소 {repo_full_name} - 유일한 GitHub 토큰이 유효하지 않습니다. 이 저장소 건너뜁니다.")
                    break # attempt 루프 종료
                continue 

            # print(f"  저장소 {repo_full_name} 처리 시도 (토큰 인덱스 {token_to_try_idx}, 토큰: {token_str_to_try[:8]}...).")
            try:
                # 현재 g 객체를 재사용할지, 아니면 token_str_to_try로 새로 만들지 결정
                needs_new_g_instance = False
                if g is None:
                    needs_new_g_instance = True
                # g.auth.token이 존재하는지, 그리고 현재 시도하려는 토큰과 다른지 확인
                elif not hasattr(g, 'auth') or g.auth is None or not hasattr(g.auth, 'token') or g.auth.token != token_str_to_try:
                    needs_new_g_instance = True
                elif g.oauth_scopes is None: # 스코프 정보가 없는 경우 (문제가 있을 수 있음)
                    needs_new_g_instance = True
                # elif g.per_page == 0: # 추가적인 유효성 검사 (선택 사항)
                #     needs_new_g_instance = True

                if needs_new_g_instance:
                    # print(f"    토큰 인덱스 {token_to_try_idx} ({token_str_to_try[:8]}...)로 GitHub 객체 (재)생성...")
                    g_temp = Github(token_str_to_try, timeout=20)
                    g_temp.get_user() # 객체 활성화 및 간단한 테스트 (RateLimit 등 여기서도 발생 가능)
                    api_object_for_repo = g_temp
                    current_token_index = token_to_try_idx # 전역 현재 토큰 인덱스 업데이트
                    g = g_temp # 전역 g 객체 업데이트
                else:
                    # print(f"    기존 GitHub 객체 재사용 (토큰 인덱스 {token_to_try_idx}, 토큰: {g.auth.token[:8]}...).")
                    api_object_for_repo = g # 기존 g 객체 재사용
                
                # 이제 api_object_for_repo를 사용
                # print(f"    토큰 인덱스 {token_to_try_idx}로 API 호출 시도...")
                repos = api_object_for_repo.get_repo(repo_full_name)
                contents = repos.get_contents("") # 루트 디렉토리 내용 가져오기
                found_setup = False
                for content_file in contents:
                    # 파일 이름만 비교 (대소문자 무시), 경로 전체 비교는 content_file.path 사용
                    if content_file.name.lower() == "setup.py":
                        setupfile_content = content_file.decoded_content.decode('utf-8', errors='replace')
                        repo_df.loc[idx, "setupfile"] = setupfile_content
                        successful_fetches += 1
                        found_setup = True
                        break 
                
                if found_setup: # 성공적으로 setup.py 찾음
                    # print(f"  성공: 저장소 {repo_full_name} setup.py 수집 완료 (토큰 인덱스 {token_to_try_idx}).")
                    break # 다음 저장소로 (현재 저장소에 대한 토큰 시도 attempt 루프 종료)

            except RateLimitExceededException:
                print(f"  경고: 저장소 {repo_full_name}, 토큰 인덱스 {token_to_try_idx} ({token_str_to_try[:8]}...) API 한도 초과. 다음 토큰 시도.")
                if attempt == len(github_tokens) - 1: # 모든 토큰 시도 후에도 한도 초과
                    print(f"  오류: 저장소 {repo_full_name} - 모든 토큰 API 한도 초과. 이 저장소 건너뜁니다.")
                    time.sleep(5) # 다음 저장소 처리 전 잠시 대기
            except UnknownObjectException: # 404 Not Found
                print(f"  정보: 저장소 {repo_full_name} (index {idx})를 찾을 수 없음 (404). 이 저장소 건너뜁니다.")
                break # 이 저장소에 대한 토큰 시도 중단, 다음 저장소로
            except (BadCredentialsException, requests.exceptions.Timeout, GithubException, requests.exceptions.ConnectionError) as e:
                print(f"  경고: 저장소 {repo_full_name}, 토큰 인덱스 {token_to_try_idx} ({token_str_to_try[:8]}...) 연결 오류 ({type(e).__name__}). 다음 토큰 시도.")
                if attempt == len(github_tokens) - 1: # 모든 토큰 시도 후에도 오류
                    print(f"  오류: 저장소 {repo_full_name} - 모든 토큰으로 연결 오류 발생. 이 저장소 건너뜁니다.")
            except Exception as e: # 예상치 못한 다른 오류
                print(f"  오류: 저장소 {repo_full_name} (index {idx}) 처리 중 예상치 못한 오류: {e} (타입: {type(e).__name__})")
                traceback.print_exc() # 상세 오류 출력
                break # 이 저장소에 대한 토큰 시도 중단, 다음 저장소로
            
            time.sleep(0.2) # 다음 토큰 시도 전 짧은 대기

        # 각 저장소 처리 후 기본 딜레이 (API 서버 부하 감소)
        time.sleep(np.random.uniform(0.3, 1.0)) 
        tiredness += 1
        if tiredness > 0 and tiredness % 100 == 0: # 일정 개수 처리 후 휴식
            rest_time = np.random.randint(20, 40) # 휴식 시간 랜덤화
            print(f'\n{tiredness}개 repository 처리 후 잠시 휴식 ({rest_time}초)...')
            time.sleep(rest_time)
    
    print(f"\n총 {successful_fetches}개의 setup.py 파일을 수집했습니다.")
    return repo_df

def double_check_lib(lib_name_input):
    if not lib_name_input or not isinstance(lib_name_input, str) or lib_name_input.strip() == "":
        return None

    lib_cleaned = lib_name_input.strip()
    url = f"https://pypi.org/project/{lib_cleaned}/"

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        page = requests.get(url, timeout=10, headers=headers)
        page.raise_for_status()
    except HTTPError: # 4xx (라이브러리 없음 등), 5xx (서버 오류)
        # print(f"정보: 라이브러리 '{lib_cleaned}' PyPI 조회 실패 (HTTP {page.status_code if 'page' in locals() else 'N/A'}).")
        return None
    except requests.exceptions.Timeout:
        print(f"경고: 라이브러리 '{lib_cleaned}' PyPI 조회 타임아웃.")
        return None
    except requests.exceptions.RequestException as e: # ConnectionError, TooManyRedirects 등
        print(f"경고: 라이브러리 '{lib_cleaned}' PyPI 조회 중 요청 에러: {e}.")
        return None
    else: # 요청 성공 시
        try:
            soup = BeautifulSoup(page.content, 'html.parser')
            description_tag = soup.find("div", class_="project-description")
            if description_tag:
                description_text = description_tag.get_text(separator=" ", strip=True).upper() # 대소문자 무시 비교 위해
                if "UNKNOWN" not in description_text and description_text: # 내용이 있고 "UNKNOWN"이 아니면 유효
                    return lib_cleaned # 원본 정리된 이름 반환
            return None # 설명 태그 없거나, 내용이 UNKNOWN이거나 비어있는 경우
        except Exception as parse_e: # 파싱 중 예외 발생 시
            print(f"경고: 라이브러리 '{lib_cleaned}' PyPI 페이지 파싱 중 에러: {parse_e}")
            return None

def add_requirelist(input_df):
    if "setupfile" not in input_df.columns:
        print("오류: 'setupfile' 컬럼이 DataFrame에 없습니다. requirement list 추출을 건너뜁니다.")
        # 'require_list' 컬럼을 만들고 None으로 채워 이후 로직에서 컬럼 부재 오류 방지
        input_df_copy = input_df.copy()
        input_df_copy["require_list"] = None
        return input_df_copy

    df = input_df.copy()
    df["setupfile_proc"] = df["setupfile"].apply(lambda x: str(x) if pd.notna(x) else "")


    re_install_requires_list = re.compile(r"""install_requires\s*=\s*\[(.*?)\]""", re.VERBOSE | re.DOTALL | re.MULTILINE)
    re_install_requires_tuple = re.compile(r"""install_requires\s*=\s*\((.*?)\)""", re.VERBOSE | re.DOTALL | re.MULTILINE)
    # extras [...] 구문까지 포함하여 라이브러리 이름 추출 시도
    re_lib_name_in_quotes = re.compile(r"""['"]\s*([^'"#;\s]+(?:\s*\[[^\]]+\])?)\s*["']""")


    def extract_libs(setup_content):
        if not isinstance(setup_content, str): return []
        
        # 라인 단위로 주석(#) 제거, 빈 라인 무시
        content_for_search = "\n".join([line.split('#')[0].strip() for line in setup_content.splitlines() if line.strip()])
        if not content_for_search: return []

        libs_str_parts = []
        match_list = re_install_requires_list.search(content_for_search)
        if match_list: libs_str_parts.append(match_list.group(1))
        
        match_tuple = re_install_requires_tuple.search(content_for_search)
        if match_tuple: libs_str_parts.append(match_tuple.group(1))
        
        if not libs_str_parts: return []
        
        full_libs_str = "\n".join(libs_str_parts)
        extracted_raw = re_lib_name_in_quotes.findall(full_libs_str)
        # 각 추출된 라이브러리 이름의 앞뒤 공백 제거
        return [lib.strip() for lib in extracted_raw if lib.strip()]

    print("1단계: setup.py에서 install_requires 내용 추출 중...")
    df["extracted_reqs"] = df["setupfile_proc"].apply(extract_libs)
    
    print("2단계: 추출된 라이브러리 이름 정리 중...")
    def clean_lib_name(raw_name_str): # PEP 508 호환성 고려하여 정리
        if not raw_name_str: return ""
        name = raw_name_str.split('@')[0].strip() # URL 부분 제거
        name = name.split(';')[0].strip()   # 환경 마커 부분 제거
        # extras 부분은 일단 유지 (예: 'requests[security]') -> double_check_lib에서 처리하거나,
        # 여기서 제거하려면: name_no_extras = re.sub(r'\[.*?\]', '', name)
        # 버전 명시자 및 기타 특수문자 제거
        name = re.split(r'[<>=!~\s(]', name, 1)[0].strip() # 첫 특수문자/공백에서 분리
        
        # PEP 426 정규화와 유사하게 처리 (소문자, 연속된 비영숫자를 단일 '-'로)
        name = name.lower()
        name = re.sub(r"[-_.]+", "-", name) # '-', '_', '.'을 모두 '-'로 통일
        name = re.sub(r"[^a-z0-9-]+", "", name) # 영숫자와 '-' 외 문자 제거
        return name.strip('-') # 앞뒤 '-' 제거

    df["cleaned_reqs"] = df["extracted_reqs"].apply(
        lambda libs: sorted(list(set(filter(None, [clean_lib_name(lib) for lib in libs])))) # 중복제거, 빈문자열제거, 정렬
    )

    print("3단계: 정리된 라이브러리 이름 PyPI에서 검증 (시간 소요)...")
    validated_reqs_list_final = []
    # tqdm의 iterable은 list로 변환하여 전체 길이를 미리 알 수 있도록 하는 것이 좋음
    for libs_to_validate in tqdm(list(df["cleaned_reqs"]), desc="Validating on PyPI"):
        current_valid_libs = []
        if libs_to_validate: # 빈 리스트가 아닐 경우에만 검증 시도
            for lib_name_candidate in libs_to_validate:
                checked_lib = double_check_lib(lib_name_candidate)
                if checked_lib:
                    current_valid_libs.append(checked_lib)
                # PyPI 요청 사이에 매우 짧은 딜레이 (선택 사항, 서버 부하 분산)
                # time.sleep(np.random.uniform(0.01, 0.03)) 
        validated_reqs_list_final.append(current_valid_libs if current_valid_libs else None) # 빈 리스트면 None
    df["require_list"] = pd.Series(validated_reqs_list_final, index=df.index)

    print("4단계: 최종 데이터 정리 및 누락된 requirement list가 있는 행 제거...")
    # 더 이상 필요 없는 중간 컬럼들 삭제
    df = df.drop(columns=["setupfile_proc", "extracted_reqs", "cleaned_reqs"], errors='ignore')
    
    rows_before_final_dropna = len(df)
    # 'require_list'가 None이거나 비어있는 (실질적으로 유효한 라이브러리가 없는) 행 제거
    df = df.dropna(subset=["require_list"]) # dropna는 None인 행을 제거
    # 추가적으로 빈 리스트도 제거하려면: df = df[df['require_list'].apply(lambda x: x is not None and len(x) > 0)]
    # 현재 로직은 double_check_lib 후 빈 리스트는 None으로 처리하므로 dropna로 충분
    print(f"유효한 'require_list'가 없는 {rows_before_final_dropna - len(df)}개의 행을 최종 제거했습니다.")
    
    return df.reset_index(drop=True)

def make_sorted_lib_df(series_of_req_lists_input):
    if not isinstance(series_of_req_lists_input, pd.Series):
        try:
            series_of_req_lists_input = pd.Series(list(series_of_req_lists_input)) # 강제 변환 시도
        except:
            print("오류: make_sorted_lib_df - 입력값을 Series로 변환할 수 없습니다.")
            return pd.DataFrame(columns=["LIBRARY", "FREQUENCY"])


    flat_list_of_all_libs = [
        lib_item 
        for req_list_sublist in series_of_req_lists_input.dropna() 
        if isinstance(req_list_sublist, list) # 각 행의 값이 리스트인지 확인
        for lib_item in req_list_sublist 
    ]
    if not flat_list_of_all_libs:
        print("분석할 라이브러리가 없습니다. 빈도수 계산을 건너뜁니다.")
        return pd.DataFrame(columns=["LIBRARY", "FREQUENCY"])
        
    lib_counts = Counter(flat_list_of_all_libs)
    sorted_libs_df = pd.DataFrame(lib_counts.items(), columns=["LIBRARY", "FREQUENCY"])
    sorted_libs_df = sorted_libs_df.sort_values(by=["FREQUENCY", "LIBRARY"], ascending=[False, True]).reset_index(drop=True)
    return sorted_libs_df

#%% Main processing flow
# 연도별 분포 시각화
if 'update_date' in data.columns:
    try:
        year_series = data['update_date'].astype(str).str.slice(0, 4)
        # 정규표현식으로 정확히 4자리 숫자만 추출하여 정수형 Nullable로 변환
        data['year'] = year_series.str.extract(r'^(\d{4})$', expand=False).astype(float).astype('Int64')
        
        valid_year_data_for_plot = data.dropna(subset=['year']) # 'year'가 NA인 행 제거
        if not valid_year_data_for_plot.empty:
            min_yr = int(valid_year_data_for_plot["year"].min())
            max_yr = int(valid_year_data_for_plot["year"].max())
            
            plt.figure(figsize=(12, 6))
            bins = range(min_yr, max_yr + 2) 
            plt.hist(valid_year_data_for_plot["year"], bins=bins, align='left', rwidth=0.85, color='darkcyan', edgecolor='black')
            plt.title('Distribution of Repositories by Update Year', fontsize=16)
            plt.xlabel('Year', fontsize=14)
            plt.ylabel('Number of Repositories', fontsize=14)
            tick_step = max(1, (max_yr - min_yr) // 20 if max_yr > min_yr else 1) # x축 눈금 간격 조절
            plt.xticks(ticks=range(min_yr, max_yr + 1, tick_step), rotation=45, ha='right')
            plt.grid(axis='y', linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else: print("연도 정보가 유효한 데이터가 없어 연도별 분포 그래프를 그릴 수 없습니다.")
    except Exception as e: print(f"연도별 분포 그래프 생성 중 오류: {e}\n{traceback.format_exc()}")
else: print("'update_date' 컬럼이 없어 연도별 분포 그래프를 그릴 수 없습니다.")


print(f"\n초기 데이터 개수 (병합 후): {len(data)}")
if 'language' in data.columns:
    data_python = data[data["language"].astype(str).str.lower() == "python"].reset_index(drop=True)
    print(f"Python 언어 필터링 후 데이터 개수: {len(data_python)}")
else:
    print("경고: 'language' 컬럼이 없습니다. Python 프로젝트 필터링을 건너뛰고 모든 데이터를 사용합니다.")
    data_python = data.copy().reset_index(drop=True) # 모든 데이터 사용 시 인덱스 리셋

if data_python.empty:
    print("처리할 Python 저장소가 없습니다. 스크립트를 종료합니다.")
    exit()

print("\nsetup.py 파일 수집 시작 (GitHub API 사용)...")
data_with_setup = add_setupfile(data_python) # 내부에서 전역 g, current_token_index 사용
# setupfile이 실제로 수집된 (None이 아닌) 데이터만 필터링
final_have_setup_data = data_with_setup.dropna(subset=["setupfile"]).reset_index(drop=True)
print(f"setup.py 파일 내용이 성공적으로 수집된 데이터 개수: {len(final_have_setup_data)}")

if final_have_setup_data.empty:
    print("setup.py 파일 내용이 있는 유효한 저장소가 없습니다. 스크립트를 종료합니다.")
    exit()

print("\nRequirement list 추출 및 검증 시작...")
data_with_requirements = add_requirelist(final_have_setup_data)
print(f"유효한 require_list가 있는 데이터 개수 (최종): {len(data_with_requirements)}")


output_dir_name = 'analysis_output_v2' # 결과 저장 디렉토리 이름 (버전 명시)
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)
    print(f"결과 저장 디렉토리 '{output_dir_name}'를 생성했습니다.")

if not data_with_requirements.empty:
    path_processed_dataframe_pkl = os.path.join(output_dir_name, f'final_processed_data_reqs_count_{len(data_with_requirements)}.pkl')
    try:
        data_with_requirements.to_pickle(path_processed_dataframe_pkl)
        print(f"최종 처리된 DataFrame을 '{path_processed_dataframe_pkl}'에 저장했습니다.")
    except Exception as e: print(f"'{path_processed_dataframe_pkl}'에 DataFrame 저장 중 오류: {e}\n{traceback.format_exc()}")
else: print("최종 처리된 데이터가 없어 저장하지 않습니다.")


if not data_with_requirements.empty and "require_list" in data_with_requirements.columns:
    print("\n라이브러리 빈도수 계산 및 저장 중...")
    final_sorted_libraries_df = make_sorted_lib_df(data_with_requirements["require_list"])
    
    if not final_sorted_libraries_df.empty:
        num_unique_libraries = len(final_sorted_libraries_df)
        path_libs_distribution_csv = os.path.join(output_dir_name, f'sorted_libraries_dist_unique_{num_unique_libraries}.csv')
        path_libs_distribution_pkl = os.path.join(output_dir_name, f'sorted_libraries_dist_unique_{num_unique_libraries}.pkl')
        try:
            final_sorted_libraries_df.to_csv(path_libs_distribution_csv, index=False, encoding='utf-8-sig')
            print(f"라이브러리 빈도수 CSV를 '{path_libs_distribution_csv}'에 저장했습니다.")
        except Exception as e: print(f"'{path_libs_distribution_csv}'에 CSV 저장 중 오류: {e}\n{traceback.format_exc()}")
        try:
            final_sorted_libraries_df.to_pickle(path_libs_distribution_pkl)
            print(f"라이브러리 빈도수 Pickle을 '{path_libs_distribution_pkl}'에 저장했습니다.")
        except Exception as e: print(f"'{path_libs_distribution_pkl}'에 Pickle 저장 중 오류: {e}\n{traceback.format_exc()}")
        
        print(f"\n가장 많이 사용된 라이브러리 Top 20 (총 {num_unique_libraries}개 중):")
        print(final_sorted_libraries_df.head(20))
    else: print("집계된 라이브러리가 없어 빈도수 데이터를 저장하지 않습니다.")
else: print("최종 데이터가 비어있거나 'require_list' 컬럼이 없어 라이브러리 빈도수 계산을 수행할 수 없습니다.")

print("\n모든 스크립트 실행이 완료되었습니다.")