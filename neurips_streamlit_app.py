import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="NeurIPS 2025 Paper Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load the NeurIPS analysis data"""
    try:
        df = pd.read_csv('neurips_2025_full_analysis.tsv', sep='\t')
        return df
    except FileNotFoundError:
        st.error("❌ Analysis file not found. Please run the comprehensive analysis first.")
        return None

@st.cache_data
def load_abstracts():
    """Load abstracts from the JSON file"""
    try:
        with open('NeurIPS 2025 Events.json', 'r') as f:
            data = json.load(f)
        
        # Create a mapping from paper name to abstract
        abstracts = {}
        for item in data:
            name = item.get('name', '')
            abstract = item.get('abstract', '')
            abstracts[name] = abstract
        
        return abstracts
    except FileNotFoundError:
        st.error("❌ NeurIPS JSON file not found.")
        return {}

@st.cache_data
def prepare_search_index(df, abstracts):
    """Prepare TF-IDF search index"""
    documents = []
    paper_ids = []
    
    for _, row in df.iterrows():
        # Combine title, keywords, theme, and abstract
        title = str(row.get('name', ''))
        keywords = str(row.get('paper_keywords', ''))
        theme = str(row.get('cluster_theme', ''))
        abstract = abstracts.get(title, '')
        
        # Create document text
        doc_text = f"{title} {keywords} {theme} {abstract}"
        documents.append(doc_text)
        paper_ids.append(row['paper_id'])
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    return vectorizer, tfidf_matrix, paper_ids

def search_papers(query, vectorizer, tfidf_matrix, paper_ids, df, top_k=50):
    """Search papers using TF-IDF similarity"""
    if not query.strip():
        return df
    
    # Transform query
    query_vector = vectorizer.transform([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get top papers
    top_indices = similarities.argsort()[::-1][:top_k]
    top_paper_ids = [paper_ids[i] for i in top_indices]
    top_scores = similarities[top_indices]
    
    # Filter dataframe and add relevance scores
    search_results = df[df['paper_id'].isin(top_paper_ids)].copy()
    
    # Add relevance scores
    score_mapping = dict(zip(top_paper_ids, top_scores))
    search_results['relevance_score'] = search_results['paper_id'].map(score_mapping)
    
    # Sort by relevance
    search_results = search_results.sort_values('relevance_score', ascending=False)
    
    return search_results

def main():
    st.title("📚 NeurIPS 2025 Paper Explorer")
    st.markdown("Explore and discover papers from NeurIPS 2025")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    abstracts = load_abstracts()
    
    # Prepare search index
    with st.spinner("Preparing search index..."):
        vectorizer, tfidf_matrix, paper_ids = prepare_search_index(df, abstracts)
    
    # Search box
    st.header("🔍 Search Papers")
    search_query = st.text_input(
        "Search by title, keywords, theme, or abstract content:",
        placeholder="e.g., neural networks, computer vision, transformers..."
    )
    
    # Apply search if query exists
    if search_query.strip():
        df = search_papers(search_query, vectorizer, tfidf_matrix, paper_ids, df)
        if len(df) == 0:
            st.warning("No papers found matching your search.")
            return
    
    # Sidebar info and filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Topic cluster filter
        cluster_themes = ['All'] + sorted(df['cluster_theme'].unique().tolist())
        selected_theme = st.selectbox("Topic Cluster Theme", cluster_themes)
        
        # Cluster ID filter
        cluster_ids = ['All'] + sorted(df['topic_cluster'].unique().tolist())
        selected_cluster_id = st.selectbox("Topic Cluster ID", cluster_ids)
        
        # Paper type filter
        paper_types = ['All'] + sorted(df['type'].unique().tolist())
        selected_type = st.selectbox("Paper Type", paper_types)
        
        st.divider()
        
        # Score threshold sliders
        st.subheader("📊 Score Thresholds")
        
        novelty_min, novelty_max = float(df['novelty_score'].min()), float(df['novelty_score'].max())
        novelty_threshold = st.slider(
            "Minimum Novelty Score", 
            min_value=novelty_min, 
            max_value=novelty_max, 
            value=novelty_min,
            step=0.1
        )
        
        impact_min, impact_max = float(df['impact_score'].min()), float(df['impact_score'].max())
        impact_threshold = st.slider(
            "Minimum Impact Score", 
            min_value=impact_min, 
            max_value=impact_max, 
            value=impact_min,
            step=0.1
        )
        
        st.divider()
        
        st.header("📊 Dataset Info")
        
        # Apply filters
        filtered_df = df.copy()
        if selected_theme != 'All':
            filtered_df = filtered_df[filtered_df['cluster_theme'] == selected_theme]
        if selected_cluster_id != 'All':
            filtered_df = filtered_df[filtered_df['topic_cluster'] == selected_cluster_id]
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['type'] == selected_type]
        
        # Apply score thresholds
        filtered_df = filtered_df[filtered_df['novelty_score'] >= novelty_threshold]
        filtered_df = filtered_df[filtered_df['impact_score'] >= impact_threshold]
        
        st.metric("Total Papers", len(df))
        st.metric("Filtered Papers", len(filtered_df))
        st.metric("Unique Clusters", df['cluster_theme'].nunique())
        st.metric("Authors", df['num_authors'].sum())
        
        # Paper types
        type_counts = filtered_df['type'].value_counts()
        st.subheader("Paper Types (Filtered)")
        for paper_type, count in type_counts.items():
            st.metric(paper_type, count)
    
    # Main content
    st.header("🔍 Paper Database")
    
    # Remove length, count, and redundant columns for display
    display_columns = [col for col in filtered_df.columns if not any(x in col.lower() for x in ['length', 'count', 'readability']) and col not in ['clean_name', 'author_list']]
    display_df = filtered_df[display_columns]
    
    # Display the filtered dataframe
    display_df_columns = display_df.columns.tolist()
    if 'relevance_score' in display_df_columns:
        # Move relevance score to front if it exists
        display_df_columns.remove('relevance_score')
        display_df_columns.insert(1, 'relevance_score')
        display_df = display_df[display_df_columns]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        column_config={
            "paper_id": st.column_config.NumberColumn("ID", width="small"),
            "relevance_score": st.column_config.NumberColumn("Relevance", format="%.3f", width="small"),
            "name": st.column_config.TextColumn("Paper Title", width="large"),
            "speakers/authors": st.column_config.TextColumn("Authors", width="medium"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "cluster_theme": st.column_config.TextColumn("Theme", width="medium"),
            "topic_cluster": st.column_config.NumberColumn("Cluster ID", width="small"),
            "combined_score": st.column_config.NumberColumn("Score", format="%.2f", width="small"),
            "novelty_score": st.column_config.NumberColumn("Novelty", format="%.2f", width="small"),
            "impact_score": st.column_config.NumberColumn("Impact", format="%.2f", width="small"),
        }
    )
    
    # Quick stats
    st.header("📈 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Combined Score", f"{filtered_df['combined_score'].mean():.2f}")
    with col2:
        st.metric("Avg Novelty Score", f"{filtered_df['novelty_score'].mean():.2f}")
    with col3:
        st.metric("Avg Impact Score", f"{filtered_df['impact_score'].mean():.2f}")
    with col4:
        st.metric("Avg Authors per Paper", f"{filtered_df['num_authors'].mean():.1f}")

if __name__ == "__main__":
    main()