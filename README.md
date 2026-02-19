📩 Spam Mail Prediction using Machine Learning
This project implements a Binary Classification model to accurately distinguish between Spam ("Bad mail") and Ham ("Good mail"). By leveraging Natural Language Processing (NLP) techniques and Logistic Regression, the system transforms raw text data into meaningful numerical features to make high-accuracy predictions.

🚀 Project Overview
The core objective was to build a simple yet powerful predictive system. The project focuses on:

Data Cleaning: Handling null values to ensure dataset integrity.

Label Encoding: Mapping categories to numerical values (Spam = 0, Ham = 1).

Feature Engineering: Utilizing TfidfVectorizer to convert text into feature vectors.

Efficient Modeling: Using Logistic Regression for fast and interpretable binary classification.

🛠️ Tech Stack
Language: Python

Libraries: * Pandas & NumPy: Data manipulation.

Scikit-learn: Model training, TF-IDF vectorization, and accuracy evaluation.

⚙️ Model Workflow
Data Preprocessing: The mail_data.csv is loaded and cleaned, replacing null values with empty strings.

Feature Extraction: Text data is transformed using TF-IDF (Term Frequency-Inverse Document Frequency), which weights words based on their importance in the message.

Train-Test Split: The data is split into 80% training and 20% testing sets using a fixed random_state for reproducibility.

Model Training: A Logistic Regression model is trained on the feature-extracted training data.

Evaluation: Training accuracy is calculated to verify that the model is learning effectively.

📊 Results
The model successfully maps engineered text features to their respective labels with high precision.

Prediction Labels: 0 for Spam, 1 for Ham.

Accuracy: The model achieves strong performance on the training data, demonstrating the power of TF-IDF in capturing spam-related keywords.

🔮 Predictive System
The repository includes a script to test custom emails. For example:

Input: "Congratulations! You won ₹50,000 cash prize..." ➡️ Output: Spam mail.

Input: "Your order #45892 has been shipped..." ➡️ Output: Ham mail.
