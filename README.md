🤖 AI Resume Screening System
========================================================================================================================================================================================================================

🌟 Project Overview
This project implements an Artificial Intelligence-powered system to automate the initial screening of job applications. It utilizes a Random Forest Classifier trained on a dataset of candidate features and recruiter decisions. The system's capability is to process a new resume (in PDF format), extract crucial information like Skills and Certifications, combine them with quantitative metrics (Experience, Projects, Salary Expectation), and predict whether the candidate should be Selected or Rejected for a specific role.

=======================================================================================================================================================================================================================
✨ Key Features :- 

[1]. PDF Text Extraction: Reads and extracts text content directly from a PDF resume file using the PyPDF2 library.

[2]. Intelligent Section Parsing: Employs regular expressions and keyword matching to accurately locate and extract content from Skills and Certifications sections.

[3]. Hybrid Feature Engineering: Combines two distinct feature types for a robust prediction:

[4]. Text Features: Skills and Certifications are converted to numerical data using TF-IDF Vectorization.

[5]. Numeric Features: Experience (Years), Projects Count, and Salary Expectation are normalized using StandardScaler.

[6]. Predictive Model: Uses a Random Forest Classifier which demonstrated high performance, achieving an accuracy of approximately 94.50% on the test set.

[7]. Interactive Screening: Guides the user through selecting a job role and inputting quantitative details to simulate a complete screening process.

=====================================================================================================================================================================================================================
🛠️ Setup and Installation
Prerequisites
Python (3.x recommended)

Jupyter Notebook or Google Colab environment.

Dependencies
Install the necessary Python packages using pip:

Bash

!pip install pandas scikit-learn numpy PyPDF2

=====================================================================================================================================================================================================================

💻 Code Architecture and Logic
The script follows a structured workflow to handle data preparation, model training, and real-time inference.

1. Data Loading and Preprocessing
Loads the CSV dataset and handles missing values.

Creates a combined text feature: df['Text_Feature'] = df['Skills'] + " " + df['Certifications'].

2. Feature TransformationText Vectorization:
   A TfidfVectorizer (with max_features=1000 and stop_words='english') is fitted to the combined text feature.
   Numeric Scaling: A StandardScaler is fitted to the numeric features.
   The transformed features are horizontally stacked (hstack) to create the final input matrix $\mathbf{X}$.

3. Model Training
The data is split using train_test_split (80% train, 20% test, with stratify=y).

A RandomForestClassifier is trained using optimized settings (n_estimators=200, max_depth=20, class_weight='balanced').

The model's performance is validated, achieving high Model Accuracy.

4. Resume Feature Extraction (Live Resume)
extract_text_from_pdf: Reads the raw text from the uploaded PDF.

clean_text: Cleans the raw text to standardize formatting.

extract_section: Uses defined keywords (SKILLS_KEYWORDS, CERT_KEYWORDS, STOP_KEYWORDS) to intelligently segment the resume and extract the relevant text blocks.

5. Prediction and ResultsNew Feature Vector Creation: The extracted text is transformed using the fitted TF-IDF Vectorizer, and the user inputs are transformed using the fitted StandardScaler.
    The new features are stacked to create the input vector $\mathbf{X}_{\text{new}}$.
    The Random Forest Model makes a prediction on $\mathbf{X}_{\text{new}}$, outputting the final Decision and the Confidence score.

=====================================================================================================================================================================================================================
▶️ How to Run the System
[1].Execute the Notebook: Run all cells in sequential order.
[2].Upload Resume: The script will prompt you to upload a PDF file for screening.
[3].Enter Details: You will be prompted to select a job role (1-4) and input the numeric details:
Years of Experience
Number of Projects
Salary Expectation ($)
[4]. View Results: The system will process the data, make a prediction, and display the final formatted PREDICTION RESULTS, including the decision, confidence, and key detected features.
