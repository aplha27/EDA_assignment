# Hiring Analysis System

## Overview
This project implements an advanced text analysis system for hiring processes, analyzing interview transcripts and resumes using natural language processing (NLP) and machine learning techniques. The system processes structured data from Excel files and extracts meaningful features to assist in hiring decisions.

## Features

### Data Processing
- Combines multiple Excel files and sheets into a unified dataset
- Handles missing values and duplicates
- Standardizes text data (lowercase conversion, name standardization)
- Removes outliers using IQR method

### Text Analysis
- **BERT Embeddings**: Utilizes DistilBERT for advanced text representation
- **Feature Extraction**:
  - Word counts and ratios
  - Text length analysis
  - Unique word percentages
  - Average word length
  - Sentiment analysis
  - Technical keyword detection

### Specific Analysis Components
- **Transcript Analysis**:
  - Filler word detection
  - Question counting
  - Response length analysis
  - Technical term identification
  - Confidence phrase detection

- **Resume Analysis**:
  - Experience extraction
  - Education level detection
  - Skills identification (technical and soft)
  - Achievement metrics
  - Format and structure analysis

### Visualization
- Correlation heatmaps
- Distribution plots
- Pair plots for feature relationships
- Text pattern analysis visualizations

 