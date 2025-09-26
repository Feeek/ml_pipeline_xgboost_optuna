from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from pandas import DataFrame
from rapidfuzz import fuzz
from tqdm import tqdm
import json
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

# Disable HDBSCAN deprecation warnings
import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


class FeatureEngineer():

    def __init__(self, dataset: DataFrame, model_name="all-MiniLM-L6-v2"):
        self.dataset = dataset
        self.model = SentenceTransformer(model_name)

        self.classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli"
        )


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


    def cluster_careers(self, categories_json: str, batch_size: int = 16):
        with open(categories_json, "r", encoding="utf-8") as f:
            categories: dict = json.load(f)

        JOB_TITLE = "job_title"
        titles = self.dataset[JOB_TITLE].unique().tolist()

        labels = list(categories.keys())
        candidate_labels = [categories[l] for l in labels]
        inv_map = {categories[l]: l for l in labels}

        ds = Dataset.from_dict({"text": titles})

        outputs = self.classifier(
            KeyDataset(ds, "text"),
            candidate_labels,
            multi_label=True,
            batch_size=batch_size,
            truncation=True
        )

        soft = {}
        for t, res in tqdm(zip(titles, outputs), total=len(titles), desc="Zero-shot scoring"):
            s = {inv_map[label]: score for label, score in zip(res["labels"], res["scores"])}
            soft[t] = s

        for lab in labels:
            col = f"p_{lab.lower()}"
            self.dataset[col] = self.dataset[JOB_TITLE].map(lambda x: soft[x][lab])


    def print_examples(self, n=30):
        sample = self.dataset.sample(n=min(n, len(self.dataset)), random_state=42)

        # znajdź kolumny, które kończą się na "_related" lub zaczynają na "p_"
        related_cols = [c for c in self.dataset.columns if c.startswith("p_")]

        for _, row in sample.iterrows():
            title = row["job_title"]
            scores = ", ".join([f"{col}={row[col]:.2f}" for col in related_cols])
            print(f"{title:<30} | {scores}")


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
