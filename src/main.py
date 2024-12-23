import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
from textblob import TextBlob
import nltk
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
file_path = "cleaned_combined_dataset_lowercase.xlsx"  # Replace with your actual file path
data = pd.read_excel(file_path)

# Drop unnecessary columns
data = data.drop(['id', 'name', 'reason_for_decision'], axis=1)

# Standardize the 'decision' column
decision_mapping = {'selected': 1, 'select': 1, 'rejected': 0, 'reject': 0}
data['decision'] = data['decision'].str.lower().map(decision_mapping)

# Verify if all values are mapped correctly
if data['decision'].isnull().any():
    raise ValueError("Some values in the 'decision' column could not be mapped. Please check the input data.")

# Handle missing values in text columns
text_columns = ['role', 'transcript', 'resume', 'job_description']
data[text_columns] = data[text_columns].fillna("")

# One-hot encode the standardized 'decision' column
onehot_encoder = OneHotEncoder(sparse_output=False)
decision_encoded = onehot_encoder.fit_transform(data[['decision']])
decision_columns = ['rejected', 'selected']  # Binary classification, use meaningful names
decision_df = pd.DataFrame(decision_encoded, columns=decision_columns)

# Add the encoded decision back to the dataset
data = data.drop('decision', axis=1)
data = pd.concat([data, decision_df], axis=1)

# Initialize DistilBERT Tokenizer and Model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to convert text to DistilBERT embeddings
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze(0).numpy()

# Convert text columns to embeddings
embedding_data = []

for col in text_columns:
    print(f"Processing {col}...")
    col_embeddings = np.stack(data[col].apply(lambda x: text_to_embedding(str(x))).to_numpy())
    embedding_data.append(pd.DataFrame(col_embeddings, columns=[f"{col}_dim_{i}" for i in range(768)]))

# Combine all embeddings into a single DataFrame
final_embeddings = pd.concat(embedding_data, axis=1)

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('words')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_hiring_features(text, text_type='transcript'):
    if not isinstance(text, str):
        text = str(text)
    
    doc = nlp(text.lower())
    
    # Common features for both transcript and resume
    common_features = {
        'word_count': len(text.split()),
        'unique_word_ratio': len(set(text.split())) / len(text.split()) if text else 0,
        'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0,
    }
    
    # Sentiment Analysis
    blob = TextBlob(text)
    sentiment_features = {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }

    if text_type == 'transcript':
        # Interview-specific features
        interview_features = {
            # Communication clarity
            'filler_words_count': len(re.findall(r'\b(um|uh|like|you know|basically|actually|literally)\b', text.lower())),
            'question_count': len(re.findall(r'\?', text)),
            'response_length_avg': sum(len(sent.split()) for sent in text.split('.')) / len(text.split('.')) if text else 0,
            
            # Technical discussion
            'technical_terms': sum(1 for token in doc if token.text in TECHNICAL_KEYWORDS),
            
            # Engagement metrics
            'interactive_phrases': len(re.findall(r'\b(could you|what if|how would|tell me|explain|describe)\b', text.lower())),
            
            # Confidence indicators
            'confidence_phrases': len(re.findall(r'\b(i know|i can|i have experience|i understand|i believe)\b', text.lower())),
            'uncertainty_phrases': len(re.findall(r'\b(maybe|perhaps|not sure|i think|possibly)\b', text.lower())),
        }
        return {**common_features, **sentiment_features, **interview_features}
    
    else:  # Resume features
        resume_features = {
            # Experience indicators
            'years_experience': len(re.findall(r'\d+[\+]?\s*(?:year|yr)s?', text.lower())),
            'num_companies': len(re.findall(r'\b(company|corporation|inc\.|ltd\.|llc)\b', text.lower())),
            
            # Education
            'education_level': sum(1 for edu in ['phd', 'master', 'bachelor', 'mba'] if edu in text.lower()),
            'gpa_mentioned': 1 if re.search(r'gpa|grade point average', text.lower()) else 0,
            
            # Skills and achievements
            'technical_skills': sum(1 for skill in TECHNICAL_SKILLS if skill in text.lower()),
            'soft_skills': sum(1 for skill in SOFT_SKILLS if skill in text.lower()),
            'achievement_words': len(re.findall(r'\b(achieved|developed|led|managed|created|improved|increased)\b', text.lower())),
            
            # Quantifiable results
            'metrics_mentioned': len(re.findall(r'\d+%|\$\d+|\d+\s*(million|thousand)', text.lower())),
            
            # Format and structure
            'bullet_points': len(re.findall(r'[•·●○*-]\s', text)),
            'sections': len(re.findall(r'\b(experience|education|skills|projects|achievements|publications)\b', text.lower())),
        }
        return {**common_features, **sentiment_features, **resume_features}

# Define keywords and skills lists
TECHNICAL_KEYWORDS = {
    'python', 'java', 'javascript', 'sql', 'aws', 'cloud', 'docker', 'kubernetes', 'machine learning',
    'ai', 'data science', 'analytics', 'algorithm', 'database', 'api', 'rest', 'git', 'agile',
    'react', 'node.js', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'sklearn'
}

TECHNICAL_SKILLS = {
    'programming', 'coding', 'software development', 'testing', 'debugging', 'deployment',
    'architecture', 'design patterns', 'data structures', 'algorithms', 'problem solving',
    'version control', 'ci/cd', 'database design', 'api development', 'system design'
}

SOFT_SKILLS = {
    'leadership', 'communication', 'teamwork', 'problem solving', 'time management',
    'project management', 'collaboration', 'adaptability', 'creativity', 'critical thinking',
    'presentation', 'negotiation', 'conflict resolution', 'mentoring'
}

# Extract features for transcript and resume
print("Extracting features from transcripts and resumes...")
transcript_features = data['transcript'].apply(lambda x: extract_hiring_features(x, 'transcript')).apply(pd.Series)
transcript_features.columns = [f'transcript_{col}' for col in transcript_features.columns]

resume_features = data['resume'].apply(lambda x: extract_hiring_features(x, 'resume')).apply(pd.Series)
resume_features.columns = [f'resume_{col}' for col in resume_features.columns]

# TF-IDF features for key terms
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
transcript_tfidf = pd.DataFrame(
    tfidf.fit_transform(data['transcript']).toarray(),
    columns=[f'transcript_tfidf_{i}' for i in range(100)]
)
resume_tfidf = pd.DataFrame(
    tfidf.fit_transform(data['resume']).toarray(),
    columns=[f'resume_tfidf_{i}' for i in range(100)]
)

# Combine all features
final_features = pd.concat([
    transcript_features,
    resume_features,
    transcript_tfidf,
    resume_tfidf,
    final_embeddings  # Keep the BERT embeddings
], axis=1)

# Feature Extraction
print("Extracting additional features...")

def extract_text_features(text):
    if not isinstance(text, str):
        text = str(text)
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text else 0,
        'sentence_count': len(text.split('.')),
        'unique_word_ratio': len(set(text.split())) / len(text.split()) if text else 0
    }

# Extract features for each text column
feature_dfs = []
for col in text_columns:
    print(f"Extracting features from {col}...")
    features = data[col].apply(extract_text_features).apply(pd.Series)
    features.columns = [f'{col}_{feat}' for feat in features.columns]
    feature_dfs.append(features)

# Combine extracted features
text_features_df = pd.concat(feature_dfs, axis=1)

# Combine all features (embeddings + extracted features)
final_embeddings = pd.concat([
    final_embeddings,
    text_features_df.reset_index(drop=True)
], axis=1)

# Apply PCA to reduce dimensions for EDA (optional)
pca = PCA(n_components=10)  # Reduce to 10 principal components
pca_embeddings = pca.fit_transform(final_embeddings)
pca_df = pd.DataFrame(pca_embeddings, columns=[f"PCA_{i+1}" for i in range(10)])

# Add encoded decision columns to reduced embeddings
final_embeddings = pd.concat([pca_df, decision_df.reset_index(drop=True)], axis=1)

# Define new combinations for EDA
combinations = [
    # Combination 1: transcript and resume features
    ['transcript_word_count', 'transcript_unique_word_ratio', 'transcript_avg_word_length',
     'resume_word_count', 'resume_unique_word_ratio', 'resume_avg_word_length'],
    
    # Combination 2: transcript, resume, and role features
    ['transcript_word_count', 'transcript_unique_word_ratio',
     'resume_word_count', 'resume_unique_word_ratio',
     'role_word_count', 'role_unique_word_ratio'],
    
    # Combination 3: All features with decision
    ['transcript_word_count', 'transcript_unique_word_ratio',
     'resume_word_count', 'resume_unique_word_ratio',
     'role_word_count', 'role_unique_word_ratio',
     'rejected', 'selected']
]

# Enhanced EDA function
def perform_detailed_eda(data, combination, title):
    subset = data[combination]
    
    print(f"\n=== {title} ===\n")
    
    # 1. Basic Statistics
    print("Summary Statistics:")
    print(subset.describe())
    print("\nCorrelation Matrix:")
    print(subset.corr())
    
    # 2. Visualizations
    plt.figure(figsize=(15, 10))
    
    # 2.1 Correlation Heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(subset.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(f"Correlation Heatmap - {title}")
    
    # 2.2 Pairplot
    plt.figure(figsize=(20, 20))
    sns.pairplot(subset, diag_kind='kde')
    plt.suptitle(f"Pairplot - {title}", y=1.02)
    
    # 2.3 Box plots if decision columns are present
    if 'selected' in combination:
        plt.figure(figsize=(15, 5))
        for col in combination:
            if col not in ['selected', 'rejected']:
                plt.figure()
                sns.boxplot(x='selected', y=col, data=subset)
                plt.title(f"{col} by Decision")
                plt.xticks([0, 1], ['Rejected', 'Selected'])
    
    plt.tight_layout()
    plt.show()

# Perform EDA for each combination
titles = [
    "Transcript and Resume Analysis",
    "Transcript, Resume, and Role Analysis",
    "Complete Feature Analysis with Decision"
]

for combination, title in zip(combinations, titles):
    perform_detailed_eda(final_features, combination, title)

# Additional analysis for text-specific patterns
def analyze_text_patterns(data):
    print("\n=== Text Pattern Analysis ===\n")
    
    # Word count distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(data['transcript_word_count'], kde=True)
    plt.title('Transcript Word Count Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(data['resume_word_count'], kde=True)
    plt.title('Resume Word Count Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(data['role_word_count'], kde=True)
    plt.title('Role Word Count Distribution')
    plt.tight_layout()
    plt.show()
    
    # Relationship between text length and decision
    if 'selected' in data.columns:
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(['transcript_word_count', 'resume_word_count', 'role_word_count']):
            plt.subplot(1, 3, i+1)
            sns.boxplot(x='selected', y=col, data=data)
            plt.title(f'{col.split("_")[0].title()} Length vs Decision')
            plt.xticks([0, 1], ['Rejected', 'Selected'])
        plt.tight_layout()
        plt.show()

# Perform text pattern analysis
analyze_text_patterns(final_features)



