# Navigating the AI Technology Landscape from GitHub Data

> **Published in** [Technology in Society](https://www.sciencedirect.com/science/article/abs/pii/S0160791X25002805), Vol. 84, Article 103090 (2026)
>
> **DOI:** [10.1016/j.techsoc.2025.103090](https://doi.org/10.1016/j.techsoc.2025.103090)
>
> **Authors:** Jaemyoung Choi, Sungsoo Lee, Hakyeon Lee
>
> **Affiliation:** Department of Industrial Engineering, Seoul National University of Science and Technology

## Abstract

인공지능(AI)이 경쟁력을 결정짓는 핵심 기술로 부상함에 따라, AI 기술의 현재와 미래를 파악하는 것이 중요해졌습니다. 기존 특허 기반 기술 분석은 등록까지의 시간 지연으로 인해 빠르게 변화하는 AI 기술을 적절히 반영하지 못하는 한계가 있습니다.

본 연구는 **GitHub 오픈소스 데이터**를 활용하여 AI 기술 지형도를 매핑하고 미래 유망 기술을 예측하는 새로운 접근법을 제시합니다. 오픈소스 소프트웨어의 코드 재사용 관계를 포착하는 **Library Coupling**이라는 새로운 계량서지학적 지표를 도입하고, **2,879개 Python 기반 AI 레포지토리**를 분석하여 **20개의 AI 기술 클러스터**를 식별하였으며, **Graph Convolutional Network(GCN) 기반 링크 예측**을 통해 AI 기술 생태계의 미래 변화를 전망합니다.

## Keywords

`Artificial Intelligence` `GitHub` `Library Coupling` `Link Prediction` `Open Source` `Technology Landscape`

## Research Framework

![Framework](https://user-images.githubusercontent.com/35715977/191416071-6eff7564-e0c4-4f0f-9484-7734468013eb.png)

### Pipeline Overview

```
1. Data Collection     GitHub API로 AI 관련 레포지토리 수집 (56,111개)
        ↓
2. Preprocessing       Python 레포 필터링 → setup.py 의존성 추출 → PyPI 검증 (2,879개)
        ↓
3. Feature Engineering  PyPI 설명 크롤링 → KeyBERT 키워드 추출 → SentenceTransformer 임베딩
        ↓
4. Network Analysis     Co-library 네트워크 + Library Coupling 네트워크 구축
        ↓
5. GNN Link Prediction  GCN 학습 → 미래 라이브러리 연결 예측 → 유망 기술 도출
```

## Project Structure

```
├── main.py                    # 메인 진입점 (NLP 분석, K-means 클러스터링, 시각화)
├── github_scraper.py          # GitHub 레포지토리 크롤러
├── prerprocessing.py          # 데이터 전처리 (Python 필터링, 의존성 추출)
├── pypi_scraper.py            # PyPI 라이브러리 설명 크롤러
├── make_node_feature.py       # 노드 피처 생성 (KeyBERT + 임베딩)
├── Make_network.py            # 네트워크 구축 (Co-library, Library Coupling)
├── make_gnn.py                # GCN 링크 예측 모델
├── fine_tuning.py             # SentenceTransformer 파인튜닝
│
├── notebook.ipynb             # 탐색적 분석 노트북
├── link_prediction.ipynb      # 링크 예측 결과 시각화
├── make_query.ipynb           # 쿼리 구성 및 데이터 탐색
│
├── material_/
│   └── crawling_material.py   # 크롤링 유틸리티 함수
│
├── data/                      # 분석 데이터 (네트워크, 피처, 결과)
├── crawled_data/              # 원본 크롤링 데이터
├── Gephi_file/                # 네트워크 시각화 파일 (GEXF/Gephi)
│
└── github_network/            # 시기별 확장 분석 (2013-2024)
    ├── data_crawling.py       # 확장 크롤러 (멀티 토큰)
    ├── library_crawling.py    # 라이브러리별 크롤링
    └── analysis_output_*/     # 시기별 분석 결과
```

## Tech Stack

| Category | Tools |
|----------|-------|
| **Data Collection** | PyGithub, BeautifulSoup, Requests |
| **NLP & Embedding** | KeyBERT, SentenceTransformer (`all-MiniLM-L6-v2`), NLTK |
| **Network Analysis** | NetworkX, Association Strength Normalization |
| **Graph Neural Network** | StellarGraph (GCN), TensorFlow/Keras |
| **Visualization** | Matplotlib, Seaborn, t-SNE, Gephi |
| **Data Processing** | Pandas, NumPy, scikit-learn |

## Key Results

- **2,879개** AI 관련 Python 레포지토리에서 **1,448개** 고유 라이브러리 식별
- Library Coupling 네트워크를 통해 **20개 AI 기술 클러스터** 도출
- GCN 기반 링크 예측으로 **미래 유망 라이브러리 조합** 발굴
- GitHub 기반 분석이 특허/논문 대비 **실시간 기술 트렌드 파악**에 효과적임을 실증

## Acknowledgement

This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Korean Government (MSIT).

## Citation

```bibtex
@article{choi2026navigating,
  title={Navigating the AI technology landscape from GitHub data},
  author={Choi, Jaemyoung and Lee, Sungsoo and Lee, Hakyeon},
  journal={Technology in Society},
  volume={84},
  pages={103090},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.techsoc.2025.103090}
}
```
