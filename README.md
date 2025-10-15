🏥 Healthcare Prediction and Consultation System

Link:[https://drive.google.com/file/d/1szv1SZCwtXEDgP1mmagW2E2M53HaTxn_/view?usp=sharing]
The Healthcare Prediction and Consultation System is an intelligent web application built using Flask and Machine Learning models to provide medical and mental health assistance.
It enables users to:

Consult about general or mental health problems

Predict potential diseases based on entered symptoms

Receive personalized consultation recommendations

🧠 Core Concept

The system combines machine learning disease prediction models with natural language–based consultation logic to simulate a basic healthcare assistant.

It uses three different machine learning algorithms to make predictions, and then performs ensemble learning (majority voting) to decide the final diagnosis.

⚙️ System Architecture

The application consists of three main layers:

1️⃣ Frontend (User Interface)

Built with HTML, CSS, and JavaScript (inside the templates/ and assets/ folders).

Each page (general.html, mental.html, prediction.html) interacts with Flask routes using AJAX/Fetch API requests.

The UI collects user input like age, gender, symptoms, mood, etc., and sends it to Flask endpoints as JSON or form data.

2️⃣ Flask Backend (Application Logic)

The backend (app.py) handles all routes and integrates the consultation and prediction logic.
Each route serves a specific function:

Route	Function
/	Home page
/general.html	Handles general medical consultation
/mental.html	Handles mental health consultation
/prediction.html	Handles disease prediction
/pred	Suggests consultation or next steps based on predicted disease
3️⃣ Machine Learning Layer (Disease Prediction)

The disease prediction part uses an ensemble of three models trained on symptom–disease datasets:

Model	Description	Role
Random Forest Classifier	Ensemble of decision trees using bagging; strong generalization and handles feature importance well.	Primary model for robust predictions
Naive Bayes Classifier	Probabilistic model using Bayes’ theorem; efficient for categorical symptom data.	Handles uncertain symptom combinations
Decision Tree Classifier	Tree-based classifier; interpretable and fast.	Provides clarity and supports ensemble diversity

All three models were trained using the same dataset containing diseases and corresponding binary-encoded symptom vectors (1 for present, 0 for absent).

Each model outputs a predicted disease label. These predictions are then combined using a majority voting mechanism implemented via SciPy’s mode() function.

🔬 Disease Prediction Logic

The function predict_disease(chosen_symptoms) executes the prediction process:

Input Encoding
Converts the user-selected symptoms into a binary vector (based on the all_symptoms_list.pkl file):

1 → symptom present

0 → symptom absent

Example:

Symptoms: ["fever", "headache"]
Vector: [1, 0, 1, 0, 0, ...]


Model Predictions
The input vector is passed to all three models:

rf_pred = random_forest_model.predict([input_vector])
nb_pred = naive_bayes_model.predict([input_vector])
dt_pred = decision_tree_model.predict([input_vector])


Majority Voting
All predictions are combined:

predictions = np.array([rf_pred, nb_pred, dt_pred])
majority_vote = mode(predictions, axis=0)[0].flatten()


Label Decoding
The encoded disease name is converted back using a Label Encoder (label_encoder.pkl):

final_disease = le.inverse_transform(majority_vote)


Final Output
The function returns the most probable disease name as the prediction result.

💬 Consultation Modules

The project integrates three types of consultation logic, each handled in consultation.py:

Module	Function	Inputs	Output
medical_consultation()	Handles general health queries	Age, Gender, Query	Text-based health advice
mental_consultation()	Handles mental health & stress analysis	Age, Gender, Stress Level, Sleep Quality, Mood, Concern	Supportive feedback or coping suggestions
predicted_consultation()	Suggests treatment or consultation specialist based on predicted disease	Disease name	Suggested doctor type or next medical step

These modules use rule-based and heuristic logic — not deep AI — but can be extended with LLM-based APIs like LangChain or GPT for conversational responses.

🧾 Model Files

All models are stored in the /models directory:

File Name	Purpose
random_forest_model.pkl	Trained Random Forest classifier
naive_bayes_model.pkl	Trained Naive Bayes classifier
decision_tree_model.pkl	Trained Decision Tree classifier
label_encoder.pkl	Label encoder mapping disease labels
all_symptoms_list.pkl	List of all symptom features used during model training
🧠 Example Workflow

User opens Prediction Page → selects symptoms.

Frontend sends JSON to /prediction.html.

Flask backend calls predict_disease() → returns disease name.

The disease name is sent to /pred → predicted_consultation() returns doctor/specialist suggestion.

Result displayed dynamically on frontend.

🧩 Future Enhancements

Integrate LLM-based consultation (LangChain or GPT API).

Add patient history tracking and user login.

Expand dataset with more diseases and symptoms.

Add confidence scores and symptom importance visualization.

Build an admin dashboard for dataset and model management.

👨‍⚕️ Summary

This project demonstrates how AI-driven disease prediction and consultation systems can work together using traditional ML models and Flask web frameworks.
It bridges the gap between rule-based healthcare guidance and data-driven prediction, forming the foundation for intelligent telemedicine systems.
