import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# --- 1. Prepare your Data (ONLY the three physics types for TRAINING) ---
# Your 'not_any_of_the_three' data will NOT be used for TRAININ
# G this model.
# It will be used for evaluating your threshold later.
data = [
    {"text": "Quantum mechanics explores subatomic particles and wave functions.", "label": "physics_type_1"},
    {"text": "Schrodinger's equation is fundamental to understanding quantum phenomena.", "label": "physics_type_1"},
    {"text": "General relativity describes gravity as spacetime curvature.", "label": "physics_type_2"},
    {"text": "Black holes and gravitational waves are predictions of Einstein's theory.", "label": "physics_type_2"},
    {"text": "High energy physics studies elementary particles and fundamental forces.", "label": "physics_type_3"},
    {"text": "The Large Hadron Collider searches for new particles like the Higgs boson.", "label": "physics_type_3"},
    {"text": "Particle accelerators are key for experimental physics research.", "label": "physics_type_3"},
    {"text": "Space-time is warped by massive objects.", "label": "physics_type_2"},
    {"text": "Quantum entanglement shows spooky action at a distance.", "label": "physics_type_1"},
    # Add more relevant data for your 3 specific physics types for robust training
]

df_known_classes = pd.DataFrame(data)
X_known = df_known_classes['text']
y_known = df_known_classes['label']

# Define the "None of the three" label explicitly for later use
NOT_ANY_LABEL = "not_any_of_the_three"

# --- 2. Encode Labels (Only for the known 3 classes) ---
label_encoder = LabelEncoder()
y_encoded_known = label_encoder.fit_transform(y_known)

# Original labels for the 3 classes that the model will learn
known_class_labels = label_encoder.classes_
print("Known class labels (for training):", known_class_labels)

# --- 3. Split Data ---
# Training only on the known 3 classes
X_train_known, X_val_known, y_train_known, y_val_known = train_test_split(
    X_known, y_encoded_known, test_size=0.2, random_state=42, stratify=y_encoded_known
)

# --- 4. Feature Extraction (TF-IDF) ---
vectorizer = TfidfVectorizer(max_df=0.85, min_df=1, ngram_range=(1, 2)) # Adjusted min_df for small example

X_train_tfidf_known = vectorizer.fit_transform(X_train_known)
X_val_tfidf_known = vectorizer.transform(X_val_known)

# --- 5. Train Logistic Regression Model (on 3 classes) ---
log_reg_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
print("\nTraining Logistic Regression model on 3 classes...")
log_reg_model.fit(X_train_tfidf_known, y_train_known)
print("Training complete.")

# --- 6. Prediction with Probability Thresholding ---

# Create a mixed evaluation set including "not_any" examples
# In a real scenario, this would be your unseen test set,
# which you know contains a mix of all 4 types.
eval_data = [
    # Known physics types (for testing model's confidence)
    {"text": "Quantum computing uses superposition for faster calculations.", "label": "physics_type_1"},
    {"text": "Gravitational lensing bends light around massive galaxies.", "label": "physics_type_2"},
    {"text": "The standard model describes fundamental particles.", "label": "physics_type_3"},
    # "Not any" examples (these are NOT trained on, but will be used to test thresholding)
    {"text": "Biological cells divide through mitosis, replicating their DNA.", "label": NOT_ANY_LABEL},
    {"text": "The human genome project mapped all genes in Homo sapiens.", "label": NOT_ANY_LABEL},
    {"text": "Algorithms and data structures are core concepts in computer science.", "label": NOT_ANY_LABEL},
    {"text": "Historical events shape societies and cultures over centuries.", "label": NOT_ANY_LABEL},
    {"text": "A new species of frog was discovered in the Amazon rainforest.", "label": NOT_ANY_LABEL},
    {"text": "Investigating ancient pottery fragments from the Roman empire.", "label": NOT_ANY_LABEL},
]
df_eval = pd.DataFrame(eval_data)
X_eval = df_eval['text']
y_eval_true_labels = df_eval['label'] # Keep true labels for final comparison

X_eval_tfidf = vectorizer.transform(X_eval) # Transform using the SAME vectorizer

# Get probabilities for the 3 known classes
probabilities = log_reg_model.predict_proba(X_eval_tfidf)
max_probabilities = np.max(probabilities, axis=1) # Get the highest probability for each sample
predicted_encoded = np.argmax(probabilities, axis=1) # Get the predicted class (0, 1, or 2)

# Define your confidence threshold
confidence_threshold = 0.7 # You will need to tune this value

# Apply the threshold logic
y_pred_final = []
for i, prob in enumerate(max_probabilities):
    if prob < confidence_threshold:
        y_pred_final.append(NOT_ANY_LABEL)
    else:
        # Convert the encoded prediction back to its original string label
        y_pred_final.append(label_encoder.inverse_transform([predicted_encoded[i]])[0])

print(f"\nPredictions with Confidence Threshold ({confidence_threshold}):")
for i, text in enumerate(X_eval):
    print(f"Text: '{text}'")
    print(f"  True Label: {y_eval_true_labels.iloc[i]}, Predicted: {y_pred_final[i]}, Max Prob: {max_probabilities[i]:.2f}")


# --- 7. Evaluation of the 4-class system (requires combining true labels) ---
# To create a classification report for the 4 classes,
# we need to ensure the `label_encoder` can handle the `NOT_ANY_LABEL` for the report.
# This means fitting it on ALL unique labels, or manually adding the mapping.

# A more robust way to evaluate the 4-class system:
# Create a combined encoder for all possible output labels (known + NOT_ANY_LABEL)
all_possible_labels = list(known_class_labels) + [NOT_ANY_LABEL]
full_label_encoder = LabelEncoder()
full_label_encoder.fit(all_possible_labels)

# Convert true and predicted labels to numerical form for the report
y_true_encoded_full = full_label_encoder.transform(y_eval_true_labels)
y_pred_encoded_full = full_label_encoder.transform(y_pred_final)

print("\nFull 4-Class Classification Report:")
print(classification_report(y_true_encoded_full, y_pred_encoded_full,
                            target_names=full_label_encoder.classes_))