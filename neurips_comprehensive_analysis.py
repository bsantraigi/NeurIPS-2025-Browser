import json
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
    
    def extract_keywords(self, text_field='clean_abstract', top_k=50):
        """Extract key terms using TF-IDF"""
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.7
        )
        
        # Fit on abstracts
        tfidf_matrix = vectorizer.fit_transform(self.df[text_field].fillna(''))
        feature_names = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Create keyword DataFrame
        keywords_df = pd.DataFrame({
            'keyword': feature_names,
            'score': mean_scores
        }).sort_values('score', ascending=False).head(top_k)
        
        return keywords_df
    
    def topic_clustering(self, n_clusters=15, text_field='clean_abstract'):
        """Perform topic clustering on papers"""
        
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
            # Fallback: TF-IDF based clustering
            print("📊 Using TF-IDF for clustering...")
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.7
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # K-means on TF-IDF vectors
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix.toarray())
            
            # Store TF-IDF matrix as embeddings for consistency
            self.embeddings = tfidf_matrix.toarray()
        
        # Add clusters to dataframe
        self.df['topic_cluster'] = clusters
        
        # Analyze clusters
        cluster_analysis = []
        for i in range(n_clusters):
            cluster_papers = self.df[self.df['topic_cluster'] == i]
            
            # Get representative keywords for this cluster
            cluster_text = ' '.join(cluster_papers[text_field].fillna(''))
            vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
            try:
                tfidf = vectorizer.fit_transform([cluster_text])
                keywords = vectorizer.get_feature_names_out()
            except:
                keywords = ['cluster_' + str(i)]
            
            cluster_analysis.append({
                'cluster': i,
                'size': len(cluster_papers),
                'keywords': list(keywords),
                'sample_titles': cluster_papers['clean_name'].head(3).tolist()
            })
        
        return pd.DataFrame(cluster_analysis)
    
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
        """Score papers based on novelty of problem statement and potential impact"""
        
        novelty_scores = []
        impact_scores = []
        
        # Keywords indicating novel problem domains
        novel_indicators = [
            'multimodal', 'cross-modal', 'few-shot', 'zero-shot', 'meta-learning',
            'emergent', 'foundation', 'large-scale', 'real-world', 'practical',
            'robust', 'adversarial', 'interpretable', 'explainable', 'fairness',
            'causal', 'quantum', 'neuromorphic', 'federated', 'privacy-preserving',
            'sustainable', 'green', 'efficient', 'lightweight', 'edge computing',
            'continual', 'lifelong', 'incremental', 'online', 'streaming'
        ]
        
        # Impact indicators (broad applicability, societal relevance)
        impact_indicators = [
            'healthcare', 'medical', 'clinical', 'diagnosis', 'treatment',
            'autonomous', 'robotics', 'navigation', 'control', 'safety',
            'climate', 'environment', 'energy', 'sustainability',
            'finance', 'economics', 'trading', 'risk',
            'education', 'learning', 'personalized', 'adaptive',
            'social', 'recommendation', 'search', 'information retrieval',
            'language', 'translation', 'communication', 'accessibility',
            'creative', 'generation', 'synthesis', 'design'
        ]
        
        for idx, row in self.df.iterrows():
            text = (row['clean_name'] + ' ' + row['clean_abstract']).lower()
            
            # Novelty score
            novelty_score = 0
            
            # 1. Novel keyword presence
            novel_matches = sum(1 for term in novel_indicators if term in text)
            novelty_score += novel_matches * 2
            
            # 2. Cross-domain indicators (combining different fields)
            domain_terms = ['vision', 'language', 'speech', 'audio', 'graph', 
                          'reinforcement', 'supervised', 'unsupervised', 'generative']
            domain_matches = sum(1 for term in domain_terms if term in text)
            if domain_matches >= 2:
                novelty_score += 3  # Cross-domain bonus
            
            # 3. Problem complexity indicators
            complexity_terms = ['multi-task', 'multi-objective', 'hierarchical', 
                              'compositional', 'modular', 'scalable']
            complexity_matches = sum(1 for term in complexity_terms if term in text)
            novelty_score += complexity_matches * 1.5
            
            # 4. Semantic novelty (if embeddings available)
            if self.embeddings is not None:
                try:
                    # Compare to cluster center
                    cluster = row['topic_cluster']
                    cluster_papers = self.df[self.df['topic_cluster'] == cluster]
                    if len(cluster_papers) > 1:
                        cluster_embeddings = self.embeddings[cluster_papers.index]
                        center = np.mean(cluster_embeddings, axis=0)
                        similarity = cosine_similarity([self.embeddings[idx]], [center])[0][0]
                        # Lower similarity to cluster center = higher novelty
                        semantic_novelty = (1 - similarity) * 5
                        novelty_score += semantic_novelty
                except Exception as e:
                    # Skip semantic novelty if there's an issue
                    pass
            
            # Impact score
            impact_score = 0
            
            # 1. Direct impact keywords
            impact_matches = sum(1 for term in impact_indicators if term in text)
            impact_score += impact_matches * 2
            
            # 2. Generalizability indicators
            general_terms = ['general', 'universal', 'unified', 'framework', 
                           'paradigm', 'foundation', 'principle']
            general_matches = sum(1 for term in general_terms if term in text)
            impact_score += general_matches * 1.5
            
            # 3. Scale indicators
            scale_terms = ['large-scale', 'massive', 'billion', 'trillion', 
                         'distributed', 'parallel', 'cloud']
            scale_matches = sum(1 for term in scale_terms if term in text)
            impact_score += scale_matches * 1.2
            
            # 4. Real-world application indicators
            real_world_terms = ['practical', 'deployment', 'production', 'industry', 
                              'commercial', 'application', 'real-world']
            rw_matches = sum(1 for term in real_world_terms if term in text)
            impact_score += rw_matches * 2
            
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
        plt.show()
        
        return fig
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        
        print("🚀 Starting comprehensive NeurIPS 2025 analysis...")
        
        # Load data
        self.load_data()
        
        # Extract keywords
        print("\n📝 Extracting keywords...")
        keywords = self.extract_keywords()
        print(f"Top 10 keywords: {', '.join(keywords.head(10)['keyword'].tolist())}")
        
        # Topic clustering
        print("\n🎯 Performing topic clustering...")
        clusters = self.topic_clustering()
        print(f"Created {len(clusters)} topic clusters")
        
        # Author analysis
        print("\n👥 Analyzing author networks...")
        author_stats = self.analyze_authors()
        print(f"Total authors: {author_stats['total_authors']}")
        print(f"Avg authors per paper: {author_stats['avg_authors_per_paper']:.2f}")
        
        # Novelty and impact assessment
        print("\n⭐ Assessing novelty and impact potential...")
        self.assess_novelty_and_impact()
        
        # Generate rankings
        print("\n🏆 Generating paper rankings...")
        rankings = self.generate_rankings()
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        self.create_visualizations()
        
        # Summary report
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📄 Total papers analyzed: {len(self.df)}")
        print(f"🎭 Unique authors: {author_stats['total_authors']}")
        print(f"🤝 Collaboration connections: {author_stats['collaboration_edges']}")
        print(f"🏷️  Top keywords: {', '.join(keywords.head(5)['keyword'].tolist())}")
        
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
            'dataframe': self.df
        }

# Usage
if __name__ == "__main__":
    analyzer = NeurIPSAnalyzer('/home/bytestorm/Downloads/neurips-2025/NeurIPS 2025 Events.json')
    results = analyzer.run_complete_analysis()
    
    # Save detailed results
    results['rankings']['best_combined'].to_csv('neurips_2025_top_papers.csv', index=False)
    results['keywords'].to_csv('neurips_2025_keywords.csv', index=False)
    
    print(f"\n💾 Detailed results saved to CSV files!")
    print("🎯 Analysis complete!")