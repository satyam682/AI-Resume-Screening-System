🤖 AI Resume Screening System
🌟 Project Overview
This project implements an Artificial Intelligence-powered system to automate the initial screening of job applications. It uses a Random Forest Classifier trained on a dataset of candidate features and recruiter decisions. The system can process a new resume (in PDF format), extract relevant information like Skills and Certifications, combine them with quantitative metrics (Experience, Projects, Salary Expectation), and predict whether the candidate should be Selected or Rejected by a recruiter for a specified role.

✨ Features
PDF Text Extraction: Uses PyPDF2 to read text content directly from a PDF resume file.

Intelligent Section Parsing: Employs regular expressions and keyword matching to specifically locate and extract the Skills and Certifications sections from the unstructured resume text.

Hybrid Feature Engineering: Combines two types of data for prediction:

Text Features: Skills and Certifications are vectorized using TF-IDF.

Numeric Features: Experience, Projects Count, and Salary Expectation are standardized using StandardScaler.

Predictive Model: A Random Forest Classifier is trained to predict the recruiter's decision with high accuracy (achieved 94.50% in the provided notebook).

User Interaction: Prompts the user to select the job role and input numeric details, offering a complete end-to-end screening simulation.

🛠️ Setup and Installation
Prerequisites
You need a Python environment, preferably using Jupyter Notebook or Google Colab where the original code was developed.

Dependencies
Install the required libraries using pip:

Bash
!pip install pandas scikit-learn numpy PyPDF2
Data Requirement
This system relies on a training dataset. Ensure you have the file AI_Resume_Screening.csv in the same directory (or path) as your notebook. This CSV file is expected to contain the following columns:

Column Name	Type	Description
Skills	Text	List of candidate skills
Certifications	Text	List of candidate certifications
Experience (Years)	Numeric	Candidate's total years of experience
Projects Count	Numeric	Number of projects listed on the resume
Salary Expectation ($)	Numeric	Candidate's expected salary
Recruiter Decision	Target	The outcome (0 for Rejected, 1 for Selected)
💻 Code Architecture and Logic
The script is logically organized into nine main sections.

1. Load & Prepare Dataset
The AI_Resume_Screening.csv is loaded.

Missing values in Skills are filled with an empty string (''), and in Certifications with 'NA'.

A combined text feature, Text_Feature, is created by concatenating Skills and Certifications.

The three numeric features are identified.

2. Text Vectorization and Feature Scaling
TF-IDF: A TfidfVectorizer is applied to the Text_Feature to convert words into a numerical representation, limited to the top 1000 features (max_features=1000).

Scaling: A StandardScaler is fitted to the numeric features (Experience, Projects Count, Salary Expectation) to normalize their scale.

Feature Combination: The text features (X_text) and the scaled numeric features (X_num) are combined horizontally using scipy.sparse.hstack to create the final feature matrix X.

3. Train Model
The data is split into training and testing sets (80/20 split, stratified on the target variable y).

A RandomForestClassifier is initialized with key hyperparameters: n_estimators=200, max_depth=20, and class_weight='balanced' (to handle potential class imbalance).

The model is trained on the training data and evaluated, achieving approximately 94.50% accuracy on the test set.

4. Extract Text from PDF
The extract_text_from_pdf function uses PyPDF2.PdfReader to iterate through all pages of the uploaded PDF file and return the aggregated raw text.

The clean_text function performs basic text normalization by replacing bullets/hyphens with newlines, removing excessive whitespace, and eliminating most special characters.

5. Extract Resume Sections
The core extraction logic resides in extract_section. It identifies a section by searching for a list of predefined section_keywords (e.g., "skills", "certifications").

The section content is defined as the text between the start keyword and the first occurrence of a stop_keywords (e.g., "education", "experience") to cleanly segment the resume.

The extracted content is further cleaned and returned.

6. Upload & Process Resume (Runtime)
This section uses google.colab.files.upload() to facilitate uploading a new PDF resume.

The text is extracted, cleaned, and then section extraction functions (for skills and certifications) are called.

The extracted lists are printed for validation.

7. Collect User Input (Runtime)
The user is prompted to select a target Job Role (e.g., Data Scientist) and input their Years of Experience, Number of Projects, and Salary Expectation.

8. Prepare Features & Predict
The newly extracted skills and certifications are joined into a single text string.

The trained TF-IDF vectorizer is used to transform the new text features (X_text_new).

The trained StandardScaler is used to transform the new numeric features (X_num_new).

All features are stacked (hstack) into the final input vector X_new.

The trained model makes the final prediction, and the confidence level (predict_proba) is calculated.

9. Display Results
The final prediction is presented clearly, showing the Applied Role, the Decision (✅ SELECTED or ❌ REJECTED), and the model's Confidence level.

The key features used for the prediction (top extracted skills, certifications, experience, and salary expectation) are also displayed.
