# Exploratory Data Analysis Report: Interview Data Analysis

## 1. Overview of Features Analyzed
### Transcript Features:
- Word count
- Unique word ratio
- Average word length

### Resume Features:
- Word count
- Unique word ratio
- Average word length

## 2. Key Statistical Findings

### 2.1 Transcript Analysis
- Word count distribution is roughly normal, centered around 500-1000 words
- Unique word ratio ranges from 0.3 to 0.8, with most values between 0.4-0.6
- Average word length is consistently between 4.5-6.0 characters

### 2.2 Resume Analysis
- Word counts typically range from 200-600 words
- Unique word ratio clusters between 0.4-0.7
- Average word length is between 5-7 characters

## 3. Correlation Analysis

### 3.1 Strong Correlations (|r| > 0.6)
- Transcript word count & unique word ratio: -0.74 (strong negative)
- Resume word count & unique word ratio: -0.67 (strong negative)

### 3.2 Moderate Correlations (0.3 < |r| < 0.6)
- Resume word count & avg word length: -0.56 (moderate negative)
- Resume unique word ratio & avg word length: 0.46 (moderate positive)

### 3.3 Weak Correlations (|r| < 0.3)
- Transcript avg word length & resume features: weak correlations overall
- Cross-document correlations are generally weak

## 4. Key Insights

### 4.1 Document Length Patterns
1. **Transcript Length**:
   - Shows wider variation than resumes
   - Most interviews contain 500-1000 words
   - Distribution is slightly right-skewed

2. **Resume Length**:
   - More concentrated distribution
   - Typically shorter than transcripts
   - Most resumes contain 200-600 words

### 4.2 Vocabulary Usage
1. **Unique Word Usage**:
   - Negative correlation with document length in both types
   - Longer documents tend to have more word repetition
   - Resumes show higher unique word ratios on average

2. **Word Length Patterns**:
   - Resumes have longer average word length
   - Less variation in transcript word length
   - Suggests more technical/formal language in resumes

## 5. Relationships Between Documents

### 5.1 Cross-Document Patterns
- Weak to moderate correlations between transcript and resume features
- Document types appear to capture different aspects of candidates
- Length of one document type doesn't strongly predict the other

### 5.2 Style Differences
- Resumes show more formal language (longer words)
- Transcripts show more natural language patterns
- Different vocabulary diversity patterns between formats

## 6. Recommendations for Analysis

1. **Feature Engineering**:
   - Consider creating composite features combining both document types
   - Normalize word counts to account for length differences
   - Develop ratio-based features for comparison

2. **Further Investigation**:
   - Analyze specific vocabulary usage
   - Examine temporal patterns in interviews
   - Investigate relationship with hiring outcomes

3. **Methodology Improvements**:
   - Consider segmentation analysis
   - Implement text quality metrics
   - Add semantic analysis features

## 7. Limitations and Considerations

1. **Data Distribution**:
   - Some outliers in both document types
   - Potential sampling biases
   - Length variations might affect feature reliability

2. **Feature Independence**:
   - Some features show high collinearity
   - May need feature selection for modeling
   - Consider dimension reduction techniques

## 8. Next Steps

1. **Additional Analysis**:
   - Sentiment analysis
   - Topic modeling
   - N-gram analysis

2. **Feature Development**:
   - Create normalized features
   - Develop domain-specific metrics
   - Consider temporal features

3. **Model Development**:
   - Feature selection based on correlations
   - Consider separate models for different document types
   - Evaluate feature importance in predictive models 