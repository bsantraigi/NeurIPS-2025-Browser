import json
import os
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple, Set
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import torch
import warnings
warnings.filterwarnings('ignore')

# Check GPU setup
def check_gpu_setup():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU detected: {device_name}")
        if 'AMD' in device_name or 'Radeon' in device_name or 'gfx' in device_name:
            print("🔥 AMD GPU with ROCm detected")
            return 'cuda'  # ROCm uses cuda interface
        else:
            print("🟢 NVIDIA GPU detected")
            return 'cuda'
    else:
        print("💻 Using CPU")
        return 'cpu'

DEVICE = check_gpu_setup()

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available, using TF-IDF fallback")

class NeurIPSAnalyzer:
    def __init__(self, json_path: str):
        """Initialize analyzer with NeurIPS data"""
        self.json_path = json_path
        self.data = None
        self.df = None
        self.embeddings = None
        self.use_embeddings = SENTENCE_TRANSFORMERS_AVAILABLE
        self.device = DEVICE
        
        if self.use_embeddings:
            try:
                # Try to use GPU first, fallback to CPU if issues
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                # Test encoding a small sample
                test_text = ["This is a test sentence."]
                _ = self.model.encode(test_text)
                print(f"✅ Using sentence transformers on {self.device.upper()}")
            except Exception as e:
                print(f"⚠️  GPU failed ({e}), falling back to CPU")
                try:
                    self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                    self.device = 'cpu'
                    print("✅ Using sentence transformers on CPU")
                except Exception as e2:
                    print(f"⚠️  Sentence transformers failed completely: {e2}")
                    self.use_embeddings = False
        
        if not self.use_embeddings:
            print("📊 Using TF-IDF for semantic analysis")
        
    def load_data(self):
        """Load and preprocess NeurIPS JSON data"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        
        # Clean paper names (remove LaTeX formatting)
        self.df['clean_name'] = self.df['name'].apply(self.clean_text)
        
        # Parse authors
        self.df['author_list'] = self.df['speakers/authors'].apply(self.parse_authors)
        self.df['num_authors'] = self.df['author_list'].apply(len)
        
        # Clean abstracts
        self.df['clean_abstract'] = self.df['abstract'].apply(self.clean_text)
        
        print(f"Loaded {len(self.df)} papers")
        return self
    
    def clean_text(self, text: str) -> str:
        """Clean LaTeX and formatting from text"""
        if not isinstance(text, str):
            return ""
        
        # Remove LaTeX commands
        text = re.sub(r'\$.*?\$', '', text)  # Remove $...$
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # Remove \command{...}
        text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove \command
        text = re.sub(r'\{|\}', '', text)  # Remove remaining braces
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of individual authors"""
        if not isinstance(author_string, str):
            return []
        
        # Split by comma and clean
        authors = [author.strip() for author in author_string.split(',')]
        return [author for author in authors if author]
    
    def extract_keywords(self, text_field='clean_abstract', top_k=50, include_phrases=True):
        """Extract key terms using TF-IDF with enhanced multi-word phrase detection"""
        
        # Create multiple vectorizers for different n-gram ranges
        single_words = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 1),
            min_df=3,
            max_df=0.7
        )
        
        multi_word_phrases = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(2, 4),  # 2-4 word phrases
            min_df=2,  # Lower threshold for phrases
            max_df=0.6,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'  # Allow alphanumeric tokens
        )
        
        # Extract text data
        texts = self.df[text_field].fillna('')
        
        # Fit vectorizers
        single_tfidf = single_words.fit_transform(texts)
        single_features = single_words.get_feature_names_out()
        single_scores = np.mean(single_tfidf.toarray(), axis=0)
        
        # Combine results
        keywords_data = []
        
        # Add single words
        for i, feature in enumerate(single_features):
            keywords_data.append({
                'keyword': feature,
                'score': single_scores[i],
                'type': 'single_word',
                'word_count': 1
            })
        
        if include_phrases:
            # Add multi-word phrases
            phrase_tfidf = multi_word_phrases.fit_transform(texts)
            phrase_features = multi_word_phrases.get_feature_names_out()
            phrase_scores = np.mean(phrase_tfidf.toarray(), axis=0)
            
            for i, feature in enumerate(phrase_features):
                # Boost score for longer phrases as they're often more specific
                word_count = len(feature.split())
                boosted_score = phrase_scores[i] * (1.2 ** (word_count - 2))
                
                keywords_data.append({
                    'keyword': feature,
                    'score': boosted_score,
                    'type': 'multi_word',
                    'word_count': word_count
                })
        
        # Create comprehensive keyword DataFrame
        keywords_df = pd.DataFrame(keywords_data)
        keywords_df = keywords_df.sort_values('score', ascending=False).head(top_k)
        
        return keywords_df
    
    def compute_text_statistics(self):
        """Compute various text statistics for each paper"""
        
        def safe_len(text):
            return len(str(text)) if pd.notna(text) else 0
        
        def safe_word_count(text):
            return len(str(text).split()) if pd.notna(text) else 0
        
        def estimate_readability(text):
            """Simple readability estimate based on sentence and word complexity"""
            if pd.isna(text) or not text:
                return 0
            
            text_str = str(text)
            sentences = text_str.count('.') + text_str.count('!') + text_str.count('?') + 1
            words = len(text_str.split())
            if sentences == 0 or words == 0:
                return 0
                
            # Simple Flesch-style approximation
            avg_sentence_length = words / sentences
            # Count syllables (very rough approximation)
            syllables = sum([max(1, len([c for c in word if c.lower() in 'aeiou'])) for word in text_str.split()])
            avg_syllables = syllables / words if words > 0 else 0
            
            # Simplified readability score (higher = more readable)
            readability = max(0, 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables))
            return round(readability, 2)
        
        # Title statistics
        self.df['title_length_chars'] = self.df['clean_name'].apply(safe_len)
        self.df['title_length_words'] = self.df['clean_name'].apply(safe_word_count)
        
        # Abstract statistics
        self.df['abstract_length_chars'] = self.df['clean_abstract'].apply(safe_len)
        self.df['abstract_length_words'] = self.df['clean_abstract'].apply(safe_word_count)
        self.df['abstract_readability'] = self.df['clean_abstract'].apply(estimate_readability)
        
        # Combined text statistics
        self.df['total_text_length'] = self.df['title_length_chars'] + self.df['abstract_length_chars']
        self.df['total_word_count'] = self.df['title_length_words'] + self.df['abstract_length_words']
        
        return self
    
    def extract_per_paper_keywords(self, top_k_per_paper=5):
        """Extract top keywords for each individual paper"""
        
        all_keywords_per_paper = []
        
        for idx, row in self.df.iterrows():
            paper_text = str(row['clean_abstract']) if pd.notna(row['clean_abstract']) else ""
            
            if not paper_text.strip():
                all_keywords_per_paper.append("")
                continue
            
            try:
                # Use TF-IDF on individual paper vs rest of corpus
                vectorizer = TfidfVectorizer(
                    max_features=50,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
                )
                
                # Create mini-corpus with this paper and some others for comparison
                other_texts = self.df['clean_abstract'].fillna('').sample(min(20, len(self.df))).tolist()
                mini_corpus = [paper_text] + other_texts
                
                tfidf_matrix = vectorizer.fit_transform(mini_corpus)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get TF-IDF scores for the target paper (first in corpus)
                paper_scores = tfidf_matrix[0].toarray().flatten()
                
                # Get top keywords for this paper
                top_indices = paper_scores.argsort()[-top_k_per_paper:][::-1]
                paper_keywords = [feature_names[i] for i in top_indices if paper_scores[i] > 0]
                
                all_keywords_per_paper.append("; ".join(paper_keywords))
                
            except Exception as e:
                # Fallback: use simple word frequency
                words = paper_text.lower().split()
                word_freq = Counter([w for w in words if len(w) > 3 and w.isalpha()])
                top_words = [word for word, count in word_freq.most_common(top_k_per_paper)]
                all_keywords_per_paper.append("; ".join(top_words))
        
        self.df['paper_keywords'] = all_keywords_per_paper
        return self
    
    def topic_clustering(self, n_clusters=15, text_field='clean_abstract'):
        """Perform topic clustering on papers with enhanced phrase-based analysis"""
        
        texts = self.df[text_field].fillna('').tolist()
        
        if self.use_embeddings:
            try:
                # Generate embeddings using detected device
                device_name = "GPU" if self.device == 'cuda' else "CPU"
                print(f"🔄 Generating embeddings on {device_name} (this may take a few minutes)...")
                self.embeddings = self.model.encode(texts, show_progress_bar=True, 
                                                  batch_size=32, device=self.device)
                
                # K-means clustering on embeddings
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(self.embeddings)
                
            except Exception as e:
                print(f"⚠️  Embedding generation failed: {e}")
                print("🔄 Falling back to TF-IDF clustering...")
                self.use_embeddings = False
                
        if not self.use_embeddings:
            # Enhanced TF-IDF based clustering with multi-word phrases
            print("📊 Using enhanced TF-IDF for clustering...")
            vectorizer = TfidfVectorizer(
                max_features=1500,
                stop_words='english',
                ngram_range=(1, 3),  # Include up to 3-word phrases
                min_df=2,  # Lower threshold to capture more technical terms
                max_df=0.7,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # K-means on TF-IDF vectors
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix.toarray())
            
            # Store TF-IDF matrix as embeddings for consistency
            self.embeddings = tfidf_matrix.toarray()
        
        # Add clusters to dataframe
        self.df['topic_cluster'] = clusters
        
        # Enhanced cluster analysis with multi-word phrase detection
        cluster_analysis = []
        for i in range(n_clusters):
            cluster_papers = self.df[self.df['topic_cluster'] == i]
            
            # Combine all text from this cluster
            cluster_text = ' '.join(cluster_papers[text_field].fillna(''))
            
            # Extract both single words and multi-word phrases for cluster characterization
            # Single words
            single_vectorizer = TfidfVectorizer(
                max_features=8, 
                stop_words='english',
                ngram_range=(1, 1),
                min_df=1
            )
            
            # Multi-word phrases  
            phrase_vectorizer = TfidfVectorizer(
                max_features=5,
                stop_words='english', 
                ngram_range=(2, 3),
                min_df=1,
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
            )
            
            try:
                # Get single word keywords
                single_tfidf = single_vectorizer.fit_transform([cluster_text])
                single_keywords = single_vectorizer.get_feature_names_out()
                
                # Get phrase keywords
                phrase_tfidf = phrase_vectorizer.fit_transform([cluster_text])
                phrase_keywords = phrase_vectorizer.get_feature_names_out()
                
                # Combine keywords, prioritizing phrases
                all_keywords = list(phrase_keywords) + list(single_keywords)
                
            except:
                all_keywords = [f'cluster_{i}']
            
            # Try to identify the dominant theme using technical phrase patterns
            cluster_theme = self._identify_cluster_theme(cluster_text)
            
            cluster_analysis.append({
                'cluster': i,
                'size': len(cluster_papers),
                'theme': cluster_theme,
                'keywords': all_keywords[:10],  # Top 10 combined keywords
                'phrases': list(phrase_keywords) if 'phrase_keywords' in locals() else [],
                'sample_titles': cluster_papers['clean_name'].head(3).tolist()
            })
        
        cluster_df = pd.DataFrame(cluster_analysis)
        
        # Map cluster themes back to the main dataframe
        cluster_theme_map = dict(zip(cluster_df['cluster'], cluster_df['theme']))
        self.df['cluster_theme'] = self.df['topic_cluster'].map(cluster_theme_map)
        
        return cluster_df
    
    def _identify_cluster_theme(self, cluster_text):
        """Identify the dominant theme of a cluster using pattern matching"""
        
        themes = {
            'Computer Vision': [
                r'\b(?:computer\s+vision|image|visual|object\s+detection|segmentation|recognition)\b',
                r'\b(?:convolutional|cnn|vision\s+transformer|vit)\b'
            ],
            'Natural Language Processing': [
                r'\b(?:natural\s+language|nlp|text|language\s+model|transformer|bert|gpt)\b',
                r'\b(?:sentiment|translation|summarization|question\s+answering)\b'
            ],
            'Reinforcement Learning': [
                r'\b(?:reinforcement\s+learning|rl|policy|reward|agent|environment)\b',
                r'\b(?:q-learning|actor-critic|deep\s+q|markov\s+decision)\b'
            ],
            'Deep Learning Architecture': [
                r'\b(?:neural\s+network|deep\s+learning|architecture|layer|activation)\b',
                r'\b(?:attention|transformer|residual|skip\s+connection)\b'
            ],
            'Optimization & Training': [
                r'\b(?:optimization|gradient|training|learning\s+rate|optimizer)\b',
                r'\b(?:convergence|backpropagation|stochastic|batch)\b'
            ],
            'Generative Models': [
                r'\b(?:generative|gan|vae|diffusion|autoencoder|generation)\b',
                r'\b(?:synthesis|sampling|latent\s+space|decoder)\b'
            ],
            'Graph Learning': [
                r'\b(?:graph|node|edge|network|topological|relational)\b',
                r'\b(?:gnn|graph\s+neural|social\s+network|knowledge\s+graph)\b'
            ],
            'Healthcare & Medicine': [
                r'\b(?:medical|clinical|healthcare|diagnosis|treatment|patient)\b',
                r'\b(?:biomedical|pharmaceutical|drug|molecular)\b'
            ],
            'Robotic & Control': [
                r'\b(?:robot|robotic|control|manipulation|navigation|autonomous)\b',
                r'\b(?:motion\s+planning|trajectory|sensor|actuator)\b'
            ],
            'Fairness & Ethics': [
                r'\b(?:bias|fairness|ethical|responsible|interpretable|explainable)\b',
                r'\b(?:algorithmic\s+fairness|trustworthy|accountability)\b'
            ]
        }
        
        cluster_text_lower = cluster_text.lower()
        theme_scores = {}
        
        for theme, patterns in themes.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, cluster_text_lower, re.IGNORECASE))
                score += matches
            theme_scores[theme] = score
        
        # Return theme with highest score, or 'Mixed' if no clear winner
        if max(theme_scores.values()) > 0:
            return max(theme_scores, key=theme_scores.get)
        else:
            return 'Mixed Topics'
    
    def compute_similarity_metrics(self):
        """Compute similarity metrics if embeddings are available"""
        
        if self.embeddings is None or not hasattr(self, 'embeddings'):
            # Add placeholder columns
            self.df['cluster_similarity'] = 0.0
            self.df['cluster_uniqueness'] = 0.0
            self.df['avg_similarity_to_others'] = 0.0
            return self
        
        cluster_similarities = []
        cluster_uniqueness_scores = []
        avg_similarities = []
        
        for idx, row in self.df.iterrows():
            try:
                paper_embedding = self.embeddings[idx].reshape(1, -1)
                cluster_id = row['topic_cluster']
                
                # 1. Similarity to cluster centroid
                cluster_papers = self.df[self.df['topic_cluster'] == cluster_id]
                if len(cluster_papers) > 1:
                    cluster_embeddings = self.embeddings[cluster_papers.index]
                    cluster_centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
                    cluster_sim = cosine_similarity(paper_embedding, cluster_centroid)[0][0]
                else:
                    cluster_sim = 1.0
                
                # 2. Uniqueness within cluster (inverse of average similarity to cluster members)
                if len(cluster_papers) > 1:
                    other_cluster_embeddings = self.embeddings[[i for i in cluster_papers.index if i != idx]]
                    if len(other_cluster_embeddings) > 0:
                        similarities_to_cluster = cosine_similarity(paper_embedding, other_cluster_embeddings)[0]
                        avg_cluster_sim = np.mean(similarities_to_cluster)
                        uniqueness = 1.0 - avg_cluster_sim
                    else:
                        uniqueness = 1.0
                else:
                    uniqueness = 1.0
                
                # 3. Average similarity to all other papers
                other_embeddings = self.embeddings[[i for i in range(len(self.embeddings)) if i != idx]]
                if len(other_embeddings) > 0:
                    all_similarities = cosine_similarity(paper_embedding, other_embeddings)[0]
                    avg_sim = np.mean(all_similarities)
                else:
                    avg_sim = 0.0
                
                cluster_similarities.append(round(cluster_sim, 4))
                cluster_uniqueness_scores.append(round(uniqueness, 4))
                avg_similarities.append(round(avg_sim, 4))
                
            except Exception as e:
                cluster_similarities.append(0.0)
                cluster_uniqueness_scores.append(0.0)
                avg_similarities.append(0.0)
        
        self.df['cluster_similarity'] = cluster_similarities
        self.df['cluster_uniqueness'] = cluster_uniqueness_scores
        self.df['avg_similarity_to_others'] = avg_similarities
        
        return self
    
    def analyze_authors(self):
        """Analyze author networks and collaborations"""
        
        # Author frequency
        all_authors = []
        for authors in self.df['author_list']:
            all_authors.extend(authors)
        
        author_counts = Counter(all_authors)
        
        # Collaboration network
        G = nx.Graph()
        
        for authors in self.df['author_list']:
            if len(authors) > 1:
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        if G.has_edge(authors[i], authors[j]):
                            G[authors[i]][authors[j]]['weight'] += 1
                        else:
                            G.add_edge(authors[i], authors[j], weight=1)
        
        # Network metrics
        top_authors = pd.DataFrame([
            {'author': author, 'paper_count': count}
            for author, count in author_counts.most_common(20)
        ])
        
        return {
            'top_authors': top_authors,
            'network': G,
            'total_authors': len(author_counts),
            'avg_authors_per_paper': self.df['num_authors'].mean(),
            'collaboration_edges': len(G.edges())
        }
    
    def assess_novelty_and_impact(self):
        """Score papers based on novelty of problem statement and potential impact using advanced pattern matching"""
        
        novelty_scores = []
        impact_scores = []
        
        # Enhanced multi-word patterns for novelty detection
        novel_patterns = {
            'emerging_paradigms': [
                r'\b(?:foundation|large language|multimodal|cross-modal)\s+models?\b',
                r'\b(?:few-shot|zero-shot|in-context)\s+learning\b',
                r'\b(?:prompt|instruction)\s+(?:engineering|tuning|learning)\b',
                r'\b(?:chain of thought|reasoning|emergent)\s+(?:prompting|abilities|behaviors?)\b',
                r'\b(?:retrieval|memory)\s+augmented\s+(?:generation|models?)\b',
                r'\b(?:constitutional|self-supervised|contrastive)\s+(?:ai|learning|training)\b'
            ],
            'novel_architectures': [
                r'\b(?:transformer|attention|self-attention)\s+(?:variants?|mechanisms?|architectures?)\b',
                r'\b(?:mixture of experts?|sparse|efficient)\s+(?:models?|architectures?|transformers?)\b',
                r'\b(?:neural|differentiable)\s+(?:odes?|programming|rendering|synthesis)\b',
                r'\b(?:graph|geometric|equivariant)\s+(?:neural networks?|deep learning|representations?)\b',
                r'\b(?:quantum|neuromorphic|spiking)\s+(?:neural networks?|computing|machine learning)\b',
                r'\b(?:capsule|dynamic|adaptive)\s+(?:networks?|routing|architectures?)\b'
            ],
            'novel_methods': [
                r'\b(?:meta|continual|lifelong|online)\s+learning\b',
                r'\b(?:adversarial|robust|certified)\s+(?:training|defense|robustness)\b',
                r'\b(?:causal|counterfactual|interventional)\s+(?:inference|reasoning|learning)\b',
                r'\b(?:privacy-preserving|federated|distributed)\s+(?:learning|training|inference)\b',
                r'\b(?:interpretable|explainable|trustworthy)\s+(?:ai|machine learning|models?)\b',
                r'\b(?:automated|neural)\s+(?:architecture search|hyperparameter optimization)\b'
            ],
            'cross_domain': [
                r'\b(?:vision|language|speech|audio|graph)\s+(?:and|plus|\+)\s+(?:language|vision|speech|audio|graph)\b',
                r'\b(?:multi-task|multi-objective|multi-domain)\s+(?:learning|optimization|training)\b',
                r'\b(?:transfer|domain)\s+(?:learning|adaptation|generalization)\b',
                r'\b(?:simulation|real-world)\s+(?:transfer|gap|adaptation)\b'
            ]
        }
        
        # Enhanced multi-word patterns for impact assessment
        impact_patterns = {
            'societal_applications': [
                r'\b(?:climate|environmental|sustainability)\s+(?:modeling|prediction|optimization)\b',
                r'\b(?:healthcare|medical|clinical)\s+(?:diagnosis|treatment|imaging|informatics)\b',
                r'\b(?:drug|molecular|protein)\s+(?:discovery|design|folding|interaction)\b',
                r'\b(?:autonomous|self-driving|intelligent)\s+(?:vehicles?|systems?|transportation)\b',
                r'\b(?:smart|intelligent)\s+(?:cities|grids?|infrastructure|manufacturing)\b',
                r'\b(?:financial|economic|market)\s+(?:modeling|prediction|analysis|trading)\b'
            ],
            'accessibility_fairness': [
                r'\b(?:bias|fairness|equity)\s+(?:detection|mitigation|evaluation)\b',
                r'\b(?:accessibility|inclusive|assistive)\s+(?:technology|ai|design)\b',
                r'\b(?:low-resource|underrepresented)\s+(?:languages?|communities|populations)\b',
                r'\b(?:algorithmic|ai|model)\s+(?:fairness|accountability|transparency)\b'
            ],
            'scalability_efficiency': [
                r'\b(?:large-scale|massively parallel|distributed)\s+(?:training|inference|computing)\b',
                r'\b(?:efficient|lightweight|mobile|edge)\s+(?:models?|architectures?|computing|deployment)\b',
                r'\b(?:green|sustainable|energy-efficient)\s+(?:ai|computing|training)\b',
                r'\b(?:real-time|low-latency|fast)\s+(?:inference|processing|optimization)\b'
            ],
            'generalization': [
                r'\b(?:general|universal|unified)\s+(?:framework|paradigm|approach|architecture)\b',
                r'\b(?:foundation|pretrained|general-purpose)\s+(?:models?|representations?)\b',
                r'\b(?:transfer|generalization|adaptation)\s+(?:learning|capabilities?|across domains)\b',
                r'\b(?:versatile|flexible|modular)\s+(?:architectures?|frameworks?|systems?)\b'
            ]
        }
        
        for idx, row in self.df.iterrows():
            text = (row['clean_name'] + ' ' + row['clean_abstract']).lower()
            
            # Novelty score calculation
            novelty_score = 0
            
            # 1. Pattern-based novelty detection
            for category, patterns in novel_patterns.items():
                category_matches = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    category_matches += matches
                
                # Weight different categories
                weights = {
                    'emerging_paradigms': 3.0,
                    'novel_architectures': 2.5, 
                    'novel_methods': 2.0,
                    'cross_domain': 2.5
                }
                novelty_score += category_matches * weights.get(category, 1.0)
            
            # 2. Technical complexity indicators
            complexity_phrases = [
                r'\b(?:multi-modal|multimodal|cross-modal)\b',
                r'\b(?:hierarchical|compositional|modular)\b',
                r'\b(?:end-to-end|jointly|simultaneously)\b',
                r'\b(?:scalable|generalizable|transferable)\b'
            ]
            
            for pattern in complexity_phrases:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                novelty_score += matches * 1.5
            
            # 3. Semantic novelty (if embeddings available)
            if self.embeddings is not None:
                try:
                    cluster = row['topic_cluster']
                    cluster_papers = self.df[self.df['topic_cluster'] == cluster]
                    if len(cluster_papers) > 1:
                        cluster_embeddings = self.embeddings[cluster_papers.index]
                        center = np.mean(cluster_embeddings, axis=0)
                        similarity = cosine_similarity([self.embeddings[idx]], [center])[0][0]
                        semantic_novelty = (1 - similarity) * 4
                        novelty_score += semantic_novelty
                except Exception:
                    pass
            
            # Impact score calculation
            impact_score = 0
            
            # 1. Pattern-based impact detection
            for category, patterns in impact_patterns.items():
                category_matches = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    category_matches += matches
                
                # Weight different categories
                weights = {
                    'societal_applications': 3.0,
                    'accessibility_fairness': 2.5,
                    'scalability_efficiency': 2.0,
                    'generalization': 2.5
                }
                impact_score += category_matches * weights.get(category, 1.0)
            
            # 2. Scale and deployment indicators
            scale_phrases = [
                r'\b(?:billion|trillion|million)\s+(?:parameters?|samples?|users?)\b',
                r'\b(?:production|deployment|industry|commercial)\s+(?:ready|scale|application)\b',
                r'\b(?:open-source|reproducible|accessible)\b'
            ]
            
            for pattern in scale_phrases:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                impact_score += matches * 2.0
            
            # 3. Collaboration and interdisciplinary indicators
            collab_phrases = [
                r'\b(?:interdisciplinary|multidisciplinary|collaborative)\b',
                r'\b(?:human-ai|human-computer|human-machine)\s+(?:interaction|collaboration|interface)\b',
                r'\b(?:social|ethical|responsible)\s+(?:ai|implications|considerations)\b'
            ]
            
            for pattern in collab_phrases:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                impact_score += matches * 1.8
            
            novelty_scores.append(novelty_score)
            impact_scores.append(impact_score)
        
        # Add scores to dataframe
        self.df['novelty_score'] = novelty_scores
        self.df['impact_score'] = impact_scores
        self.df['combined_score'] = np.array(novelty_scores) + np.array(impact_scores)
        
        return self
    
    def generate_rankings(self, top_k=50):
        """Generate top papers by different criteria"""
        
        rankings = {
            'most_novel': self.df.nlargest(top_k, 'novelty_score')[
                ['clean_name', 'novelty_score', 'impact_score', 'combined_score', 'speakers/authors']
            ],
            'highest_impact': self.df.nlargest(top_k, 'impact_score')[
                ['clean_name', 'novelty_score', 'impact_score', 'combined_score', 'speakers/authors']
            ],
            'best_combined': self.df.nlargest(top_k, 'combined_score')[
                ['clean_name', 'novelty_score', 'impact_score', 'combined_score', 'speakers/authors']
            ]
        }
        
        return rankings
    
    def create_visualizations(self):
        """Create analysis visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Topic cluster distribution
        cluster_counts = self.df['topic_cluster'].value_counts().sort_index()
        axes[0, 0].bar(cluster_counts.index, cluster_counts.values)
        axes[0, 0].set_title('Papers per Topic Cluster')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Number of Papers')
        
        # 2. Novelty score distribution
        axes[0, 1].hist(self.df['novelty_score'], bins=30, alpha=0.7, color='blue')
        axes[0, 1].set_title('Novelty Score Distribution')
        axes[0, 1].set_xlabel('Novelty Score')
        axes[0, 1].set_ylabel('Number of Papers')
        
        # 3. Impact score distribution
        axes[0, 2].hist(self.df['impact_score'], bins=30, alpha=0.7, color='green')
        axes[0, 2].set_title('Impact Score Distribution')
        axes[0, 2].set_xlabel('Impact Score')
        axes[0, 2].set_ylabel('Number of Papers')
        
        # 4. Authors per paper
        axes[1, 0].hist(self.df['num_authors'], bins=range(1, 15), alpha=0.7)
        axes[1, 0].set_title('Authors per Paper Distribution')
        axes[1, 0].set_xlabel('Number of Authors')
        axes[1, 0].set_ylabel('Number of Papers')
        
        # 5. Novelty vs Impact scatter
        axes[1, 1].scatter(self.df['novelty_score'], self.df['impact_score'], 
                          alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Novelty Score')
        axes[1, 1].set_ylabel('Impact Score')
        axes[1, 1].set_title('Novelty vs Impact Potential')
        
        # 6. Paper type distribution
        type_counts = self.df['type'].value_counts()
        axes[1, 2].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[1, 2].set_title('Paper Type Distribution')
        
        plt.tight_layout()
        plt.savefig('neurips_2025_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        return fig
    
    def export_full_dataframe(self, filename='neurips_2025_full_analysis.tsv'):
        """Export comprehensive dataframe with all computed features to TSV"""
        
        # Define the columns we want to export in a logical order
        export_columns = [
            # Core paper information
            'name',                          # Original paper title
            'clean_name',                    # Cleaned paper title
            'speakers/authors',              # Original authors string
            'author_list',                   # Parsed authors list
            'num_authors',                   # Number of authors
            'type',                          # Paper type
            
            # Text statistics
            'title_length_chars',            # Title length in characters
            'title_length_words',            # Title length in words
            'abstract_length_chars',         # Abstract length in characters
            'abstract_length_words',         # Abstract length in words
            'abstract_readability',          # Readability score
            'total_text_length',             # Combined title + abstract characters
            'total_word_count',              # Combined title + abstract words
            
            # Keywords and content analysis
            'paper_keywords',                # Top keywords for this specific paper
            
            # Topic clustering
            'topic_cluster',                 # Assigned cluster ID
            'cluster_theme',                 # Theme of the cluster
            
            # Similarity metrics (if available)
            'cluster_similarity',            # Similarity to cluster centroid
            'cluster_uniqueness',            # Uniqueness within cluster
            'avg_similarity_to_others',      # Average similarity to all other papers
            
            # Scoring and ranking
            'novelty_score',                 # Novelty assessment score
            'impact_score',                  # Impact potential score
            'combined_score',                # Combined novelty + impact score
        ]
        
        # Check which columns actually exist in the dataframe
        available_columns = [col for col in export_columns if col in self.df.columns]
        missing_columns = [col for col in export_columns if col not in self.df.columns]
        
        if missing_columns:
            print(f"⚠️  Warning: Missing columns in export: {missing_columns}")
        
        # Create export dataframe
        export_df = self.df[available_columns].copy()
        
        # Convert list columns to string representation for TSV export
        for col in export_df.columns:
            if export_df[col].dtype == 'object':
                export_df[col] = export_df[col].apply(lambda x: str(x) if x is not None and not (isinstance(x, float) and pd.isna(x)) else '')
        
        # Add paper index as first column
        export_df.insert(0, 'paper_id', range(1, len(export_df) + 1))
        
        # Sort by combined score (highest first) for easier browsing
        if 'combined_score' in export_df.columns:
            export_df = export_df.sort_values('combined_score', ascending=False)
        
        # Export to TSV
        export_df.to_csv(filename, sep='\t', index=False, encoding='utf-8')
        
        print(f"💾 Full analysis exported to {filename}")
        print(f"📊 Exported {len(export_df)} papers with {len(export_df.columns)} features")
        print(f"📁 File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
        
        # Print column summary
        print("\n📋 Exported columns:")
        for i, col in enumerate(export_df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        return export_df
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline with enhanced multi-word phrase support"""
        
        print("🚀 Starting comprehensive NeurIPS 2025 analysis with enhanced phrase detection...")
        
        # Load data
        self.load_data()
        
        # Extract keywords with multi-word phrases
        print("\n📝 Extracting keywords and multi-word phrases...")
        keywords = self.extract_keywords(include_phrases=True)
        print(f"Top 10 keywords/phrases: {', '.join(keywords.head(10)['keyword'].tolist())}")
        
        # Topic clustering with enhanced phrase analysis
        print("\n🎯 Performing enhanced topic clustering...")
        clusters = self.topic_clustering()
        print(f"Created {len(clusters)} topic clusters with thematic analysis")
        
        # Author analysis
        print("\n👥 Analyzing author networks...")
        author_stats = self.analyze_authors()
        print(f"Total authors: {author_stats['total_authors']}")
        print(f"Avg authors per paper: {author_stats['avg_authors_per_paper']:.2f}")
        
        # Enhanced novelty and impact assessment
        print("\n⭐ Assessing novelty and impact with advanced pattern matching...")
        self.assess_novelty_and_impact()
        
        # Compute additional features for comprehensive export
        print("\n📏 Computing text statistics...")
        self.compute_text_statistics()
        
        print("\n🔍 Extracting per-paper keywords...")
        self.extract_per_paper_keywords()
        
        print("\n📐 Computing similarity metrics...")
        self.compute_similarity_metrics()
        
        # Generate rankings
        print("\n🏆 Generating paper rankings...")
        rankings = self.generate_rankings()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        self.create_visualizations()
        
        # Export comprehensive TSV
        print("\n💾 Exporting comprehensive analysis to TSV...")
        full_df = self.export_full_dataframe()
        
        # Enhanced summary report
        print("\n" + "="*60)
        print("📊 ENHANCED ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📄 Total papers analyzed: {len(self.df)}")
        print(f"🎭 Unique authors: {author_stats['total_authors']}")
        print(f"🤝 Collaboration connections: {author_stats['collaboration_edges']}")
        
        # Show keyword types breakdown
        single_word_count = len(keywords[keywords['type'] == 'single_word'])
        multi_word_count = len(keywords[keywords['type'] == 'multi_word'])
        print(f"🏷️  Keywords: {single_word_count} single-word, {multi_word_count} multi-word phrases")
        
        # Show top keywords by type
        print(f"📝 Top single words: {', '.join(keywords[keywords['type'] == 'single_word'].head(3)['keyword'].tolist())}")
        if multi_word_count > 0:
            print(f"🔤 Top multi-word phrases: {', '.join(keywords[keywords['type'] == 'multi_word'].head(3)['keyword'].tolist())}")
        
        # Show cluster themes
        print(f"\n🎯 Topic clusters by theme:")
        for _, cluster in clusters.iterrows():
            print(f"  Cluster {cluster['cluster']}: {cluster['theme']} ({cluster['size']} papers)")
        
        print(f"\n🌟 Novelty scores - Mean: {self.df['novelty_score'].mean():.2f}, Max: {self.df['novelty_score'].max():.2f}")
        print(f"🚀 Impact scores - Mean: {self.df['impact_score'].mean():.2f}, Max: {self.df['impact_score'].max():.2f}")
        
        print("\n🏆 TOP 5 MOST NOVEL PAPERS:")
        for idx, row in rankings['most_novel'].head().iterrows():
            print(f"  • {row['clean_name'][:80]}... (Score: {row['novelty_score']:.1f})")
        
        print("\n🚀 TOP 5 HIGHEST IMPACT POTENTIAL:")
        for idx, row in rankings['highest_impact'].head().iterrows():
            print(f"  • {row['clean_name'][:80]}... (Score: {row['impact_score']:.1f})")
        
        print("\n⭐ TOP 5 BEST COMBINED (NOVEL + IMPACT):")
        for idx, row in rankings['best_combined'].head().iterrows():
            print(f"  • {row['clean_name'][:80]}... (Score: {row['combined_score']:.1f})")
        
        return {
            'keywords': keywords,
            'clusters': clusters,
            'author_stats': author_stats,
            'rankings': rankings,
            'dataframe': self.df,
            'full_export': full_df
        }

# Usage
if __name__ == "__main__":
    analyzer = NeurIPSAnalyzer('/home/bytestorm/Downloads/neurips-2025/NeurIPS 2025 Events.json')
    results = analyzer.run_complete_analysis()
    
    # Save detailed results
    results['rankings']['best_combined'].to_csv('neurips_2025_top_papers.csv', index=False)
    results['keywords'].to_csv('neurips_2025_keywords.csv', index=False)
    
    print(f"\n💾 Detailed results saved to CSV files!")
    print("🎯 Enhanced analysis complete!")