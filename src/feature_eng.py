from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
from rapidfuzz import fuzz
from tqdm import tqdm
import json
from collections import Counter
import numpy as np

# Disable HDBSCAN deprecation warnings
import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


class FeatureEngineer():

    def __init__(self, dataset: DataFrame, model_name="all-MiniLM-L6-v2"):
        self.dataset = dataset
        self.model = SentenceTransformer(model_name)
        self.classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-deberta-v3-base")


    def cleanup(self):
        self.dataset = self.dataset.drop(columns=["salary"])
        
        LEVEL_WORDS = [
            "junior", "trainee"
            "graduate", "entry",
            "mid", "regular",
            "senior", "expert",
        ]

        self.dataset["job_title"] = self._get_stripped_column("job_title", LEVEL_WORDS)


    def prepare(self):
        JOB_TITLE = 'job_title'

        self._add_research_column()


        titles = self.dataset[JOB_TITLE].unique().tolist()
        embeddings = self.model.encode(titles, show_progress_bar=False)
        freq = self.dataset[JOB_TITLE].value_counts().to_dict()

        def tokens_almost_equal(t1: str, t2: str, threshold=90) -> bool:
            return t1 == t2 or fuzz.ratio(t1, t2) >= threshold

        def titles_equivalent(title1: str, title2: str, threshold=90) -> bool:
            tokens1 = title1.lower().split()
            tokens2 = title2.lower().split()

            if len(tokens1) != len(tokens2):
                return False

            return all(tokens_almost_equal(a, b, threshold) for a, b in zip(tokens1, tokens2))

        mapping = {}
        for i in tqdm(range(len(titles)), desc="Merging fuzzy duplicates"):
            for j in range(i + 1, len(titles)):
                if titles_equivalent(titles[i], titles[j], threshold=90):
                    if freq.get(titles[i], 0) >= freq.get(titles[j], 0):
                        mapping[titles[j]] = titles[i]
                    else:
                        mapping[titles[i]] = titles[j]

        self.dataset[JOB_TITLE] = self.dataset[JOB_TITLE].replace(mapping)


        titles = self.dataset[JOB_TITLE].unique().tolist()
        embeddings = self.model.encode(titles, show_progress_bar=False)
        freq = self.dataset[JOB_TITLE].value_counts().to_dict()

        mapping = {}
        for i in tqdm(range(len(titles)), desc="Merging exact duplicates"):
            for j in range(i + 1, len(titles)):
                tokens_i = set(titles[i].lower().split())
                tokens_j = set(titles[j].lower().split())
                overlap = len(tokens_i & tokens_j) / max(len(tokens_i), len(tokens_j))

                if overlap == 1.0:
                    if freq.get(titles[i], 0) >= freq.get(titles[j], 0):
                        mapping[titles[j]] = titles[i]
                    else:
                        mapping[titles[i]] = titles[j]

        self.dataset[JOB_TITLE] = self.dataset[JOB_TITLE].replace(mapping)


        titles = self.dataset[JOB_TITLE].unique().tolist()
        embeddings = self.model.encode(titles, show_progress_bar=False)

        sim_matrix = cosine_similarity(embeddings)
        title2flag = self.dataset.groupby(JOB_TITLE)["research_related"].first().to_dict()
        freq = self.dataset[JOB_TITLE].value_counts().to_dict()

        mapping = {}
        for i in tqdm(range(len(titles)), desc="Merging synonyms of titles"):
            for j in range(i + 1, len(titles)):
                if sim_matrix[i, j] >= 0.95 and title2flag[titles[i]] == title2flag[titles[j]]:
                    if freq[titles[i]] >= freq[titles[j]]:
                        mapping[titles[j]] = titles[i]
                    else:
                        mapping[titles[i]] = titles[j]

        self.dataset[JOB_TITLE] = self.dataset[JOB_TITLE].replace(mapping)

    def _add_research_column(self):
        def is_about_research(text):
            return fuzz.partial_ratio("research", str(text).lower()) >= 80

        self.dataset['research_related'] = self.dataset["job_title"].apply(is_about_research)

    def cluster_careers(self, categories_json: str, threshold: float = 0.5):
        with open(categories_json, "r", encoding="utf-8") as f:
            categories: dict = json.load(f)

        JOB_TITLE = "job_title"
        titles = self.dataset[JOB_TITLE].unique().tolist()

        # budujemy etykiety rozszerzone o warianty
        candidate_labels = list(categories.values())
        label2cat = {v: k for k, v in categories.items()}

        mapping = {}
        confidences = {}

        for t in tqdm(titles, desc="Zero-shot classifying"):
            result = self.classifier(t, candidate_labels, multi_label=False)
            best_label = result["labels"][0]
            best_score = result["scores"][0]

            if best_score < threshold:
                mapping[t] = "Other"
            else:
                mapping[t] = label2cat[best_label]   # mapujemy na główną kategorię

            confidences[t] = float(best_score)

        self.dataset["career_group"] = self.dataset[JOB_TITLE].map(mapping)
        self.dataset["career_confidence"] = self.dataset[JOB_TITLE].map(confidences)



    def print_clusters(self, examples_per_group=3):
        import random

        groups = self.dataset.groupby("career_group")["job_title"].unique()

        for group_name, titles in groups.items():
            sample_titles = random.sample(list(titles), min(examples_per_group, len(titles)))
            print(f"\nGrupa: {group_name}")
            for t in sample_titles:
                print(f"  - {t}")
            if len(titles) > examples_per_group:
                print(f"  ... i {len(titles) - examples_per_group} więcej")


    def _get_stripped_column(self, column: str, words: list[str]):
        embs_levels = self.model.encode(words, normalize_embeddings=True)

        vocab = set(" ".join(self.dataset[column].astype(str)).split())
        vocab_embs = self.model.encode(list(vocab), normalize_embeddings=True)
        emb_cache = dict(zip(vocab, vocab_embs))

        def strip_level_words(title: str) -> str:
            kept = []
            for w in title.split():
                sims = cosine_similarity([emb_cache[w]], embs_levels)[0]
                if sims.max() < 0.8:
                    kept.append(w)
            
            return " ".join(kept) if kept else title

        unique_titles = self.dataset[column].astype(str).unique()
        mapping = {t: strip_level_words(t) for t in unique_titles}

        return self.dataset[column].replace(mapping)
