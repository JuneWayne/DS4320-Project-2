# DS4320 Project 2 - Pipeline

## Loading the data


```python
import os
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Configure logging to track progress and errors
logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Retrieve MongoDB password from environment variables
mongo_pwd = os.getenv("MONGO_PWD")
# Build the MongoDB connection URL with credentials
mongo_uri = f"mongodb+srv://ethan128_db_user:{mongo_pwd}@clusterlarge.hjylasg.mongodb.net/ai_detection"

# Connect to MongoDB database
try:
    client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
    db = client["ai_detection"]
    logging.info("connected to MongoDB")
    print("connected")
except Exception as e:
    logging.error(f"connection failed: {e}")
    raise

# Query database to get all text documents
cursor = db["texts"].find({}, {
    "_id": 0,
    "attack": 0
})

# Convert database results into a pandas DataFrame for easier manipulation
df = pd.DataFrame(list(cursor))
logging.info(f"loaded {len(df)} documents from MongoDB")
print(f"shape: {df.shape}")
df.head()
```

    connected
    shape: (16000, 13)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>raid_id</th>
      <th>model_id</th>
      <th>domain_id</th>
      <th>text</th>
      <th>label</th>
      <th>decoding</th>
      <th>repetition_penalty</th>
      <th>title</th>
      <th>word_count</th>
      <th>avg_sentence_length</th>
      <th>unique_word_ratio</th>
      <th>avg_word_length</th>
      <th>punctuation_density</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>832db762-c9a8-43bd-bb69-5d2f12eae88a</td>
      <td>mistral-chat</td>
      <td>abstracts</td>
      <td>In this paper, we&nbsp;&nbsp; investigate the use of unc...</td>
      <td>1</td>
      <td>sampling</td>
      <td>yes</td>
      <td>Utilizing Uncertainty Estimation in Deep Learn...</td>
      <td>96</td>
      <td>24.000000</td>
      <td>0.875000</td>
      <td>6.322917</td>
      <td>0.009709</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28eb6f3d-fce9-4c2e-abe5-b3d436ad7680</td>
      <td>mistral</td>
      <td>abstracts</td>
      <td>http://arxiv.org/abs/1507.02843 Direct imaging...</td>
      <td>1</td>
      <td>sampling</td>
      <td>yes</td>
      <td>Direct Imaging of Nanoscale Conductance Evolut...</td>
      <td>137</td>
      <td>10.538462</td>
      <td>0.824818</td>
      <td>5.912409</td>
      <td>0.019527</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c907cb98-865b-40a8-a591-7b824d12cd57</td>
      <td>mpt-chat</td>
      <td>abstracts</td>
      <td>The Edge Clique Cover problem is a well studie...</td>
      <td>1</td>
      <td>sampling</td>
      <td>yes</td>
      <td>Edge Clique Cover of Claw-free Graphs</td>
      <td>118</td>
      <td>23.600000</td>
      <td>0.889831</td>
      <td>5.737288</td>
      <td>0.010076</td>
    </tr>
    <tr>
      <th>3</th>
      <td>931e0b06-6af3-44f0-9c78-2e6f3c39b150</td>
      <td>mpt-chat</td>
      <td>abstracts</td>
      <td>This academic paper presents&nbsp;&nbsp;a novel shape-re...</td>
      <td>1</td>
      <td>greedy</td>
      <td>no</td>
      <td>Automatic 3D bi-ventricular segmentation of ca...</td>
      <td>82</td>
      <td>20.500000</td>
      <td>0.719512</td>
      <td>6.512195</td>
      <td>0.006339</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fbecaf59-8488-45a7-b6bc-b3ccfcca1500</td>
      <td>mpt</td>
      <td>abstracts</td>
      <td>1. IntroductionConvNets have been dominating T...</td>
      <td>1</td>
      <td>sampling</td>
      <td>yes</td>
      <td>HS-ResNet: Hierarchical-Split Block on Convolu...</td>
      <td>401</td>
      <td>50.125000</td>
      <td>0.997506</td>
      <td>6.523691</td>
      <td>0.004642</td>
    </tr>
  </tbody>
</table>
</div>



## Feature Engineering

## Feature Engineering

To build the feature matrix, I combined two types of features: TF-IDF vectors pulled from the raw text, and a small set of linguistic features I computed by hand from each document.

For the TF-IDF side of things, I used sklearn's `TfidfVectorizer` with a few specific settings. First, `max_features=5000` so I'm only keeping the top 5000 most useful tokens rather than every single word in the corpus. Second, `ngram_range=(1, 2)` so the vectorizer picks up not just single words but also two-word phrases like "it is" or "in conclusion", these are the kind of phrases that tend to pop up a lot in AI writing. And third, `sublinear_tf=True` to log-normalize the term frequency, which basically just tones down the effect of words that appear way too frequently to be useful. I also made sure to fit the vectorizer only on the training set and then transform the test set separately, so no vocabulary from the test documents sneaks into the model during training.

On top of that, I added six simple statistics I computed directly from the raw text: word count, average sentence length, unique word ratio (which is basically a measure of how varied the vocabulary is), average word length, punctuation density, and the encoded content domain. The reasoning behind adding these is that they capture more general stylistic patterns that TF-IDF alone might miss. for example, AI text tends to use longer and more formal words compared to human writing, which shows up clearly in the average word length feature.

Once I had both sets of features ready, I converted them both to sparse matrices and stacked them side by side using `scipy.sparse.hstack`, ending up with a final feature matrix of shape `(n_samples, 5006)`.


```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

# Create encoders to convert categorical values into numbers that the model can understand
le_domain = LabelEncoder()
le_decoding = LabelEncoder()

# Convert domain names (like "abstracts", "forums") into integer codes
df["domain_enc"] = le_domain.fit_transform(df["domain_id"].astype(str))

# Fill missing decoding values with "none" for human-written texts, then encode
df["decoding"] = df["decoding"].fillna("none")
df["decoding_enc"] = le_decoding.fit_transform(df["decoding"].astype(str))

# Encode whether text used repetition penalty (yes/no becomes 1/0, missing becomes -1)
df["rep_penalty_enc"] = df["repetition_penalty"].map({"yes": 1, "no": 0}).fillna(-1)

# List of features that will be used by the model for classification
feature_cols = [
    "word_count",
    "avg_sentence_length",
    "unique_word_ratio",
    "avg_word_length",
    "punctuation_density",
    "domain_enc",
]

# Separate out the test set with unseen AI models to evaluate generalization
test_models = {"llama-chat"}
ai_test_mask = df["model_id"].isin(test_models)
ai_train_mask = (~ai_test_mask) & (df["label"] == 1)
human_mask = df["label"] == 0

# Split human texts between train and test sets to ensure humans are in both
human_idx = df[human_mask].index
human_train_idx, human_test_idx = train_test_split(
    human_idx, test_size=0.2, random_state=42
)

# Combine AI and human indices to create training and testing datasets
train_val_idx = df[ai_train_mask].index.tolist() + human_train_idx.tolist()
test_idx = df[ai_test_mask].index.tolist() + human_test_idx.tolist()

train_val_df = df.loc[train_val_idx]
test_df = df.loc[test_idx]

# Convert text documents into numerical vectors using TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf_train = tfidf.fit_transform(train_val_df["text"].astype(str))
X_tfidf_test = tfidf.transform(test_df["text"].astype(str))

# Convert handcrafted features into sparse matrix format for efficiency
X_hand_train = sp.csr_matrix(train_val_df[feature_cols].values)
X_hand_test = sp.csr_matrix(test_df[feature_cols].values)

# Combine TF-IDF vectors and handcrafted features into one feature matrix
X_train_full = sp.hstack([X_tfidf_train, X_hand_train])
X_test = sp.hstack([X_tfidf_test, X_hand_test])

# Extract the labels (0 for human, 1 for AI)
y_train_full = train_val_df["label"].values
y_test = test_df["label"].values

# Further split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.125, random_state=42, stratify=y_train_full
)

logging.info(f"train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")
print(f"train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")
print(f"feature matrix shape: {X_train_full.shape}")
print(f"train class | human: {(y_train==0).sum()}, AI: {(y_train==1).sum()}")
print(f"val class | human: {(y_val==0).sum()}, AI: {(y_val==1).sum()}")
print(f"test class | human: {(y_test==0).sum()}, AI: {(y_test==1).sum()}")
```

    train: 11331, val: 1619, test: 3050
    feature matrix shape: (12950, 5006)
    train class | human: 1007, AI: 10324
    val class | human: 144, AI: 1475
    test class | human: 288, AI: 2762


## Building the Logistic Regression

## Model Optimization

For the classifier, I went with Logistic Regression since it works really well with sparse high-dimensional input like TF-IDF, and it trains fast enough to run on CPU without needing a GPU.

The main challenge I ran into was the severe class imbalance in the dataset. Roughly 91% of the samples are AI-generated and only about 9% are human-written. Because of this, a vanilla logistic regression would just default to predicting AI for everything and still get a high accuracy, which is not actually useful. To get around this, I added a `class_weight` parameter that tells the model to penalize misclassifying a human sample more heavily than misclassifying an AI sample.

The tricky part was figuring out how much extra weight to give the human class, and also how much regularization to apply via the `C` parameter. Too little regularization and the model overfits on the training vocabulary, too much and it stops learning meaningful patterns altogether. So instead of guessing, I ran a grid search over human class weights `[1, 3, 5, 7, 10, 15, 20]` and C values `[0.01, 0.05, 0.1, 0.5, 1.0]`, training a separate model for each combination and evaluating it on the validation set. I used PR-AUC as the evaluation metric here rather than accuracy or ROC-AUC, since PR-AUC is more informative when the classes are this imbalanced. It focuses specifically on how well the model identifies the minority class across different decision thresholds.

The best combination from the search turned out to be `class_weight={0: 20, 1: 1}` with `C=1.0`, meaning the model penalizes human misclassifications 20 times more heavily than AI misclassifications. I then retrained the final model using these parameters on the full training set before evaluating on the held-out test set.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import numpy as np

# Define different class weights to balance the imbalance between human and AI texts since human
# class is severely imbalanced
weights = [1, 3, 5, 7, 10, 15, 20]
# Define different regularization values to control how strictly the model fits
C_values = [0.01, 0.05, 0.1, 0.5, 1.0]

# Track the best performance found so far
best_pr_auc = 0
best_params = {}
results = []

# Try all combinations of weights and regularization values
for w in weights:
    for c in C_values:
        # Create a model with current settings
        m = LogisticRegression(
            class_weight={0: w, 1: 1},
            max_iter=5000,
            random_state=42,
            solver="liblinear",
            C=c,
        )
        # Train the model on the training data
        m.fit(X_train, y_train)

        # Test on validation set and calculate performance
        val_proba = m.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_proba)
        results.append({"weight": w, "C": c, "pr_auc": pr_auc})

        # Keep track of the best performing settings
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_params = {"weight": w, "C": c}

# Show all results sorted by performance (best first)
results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print(results_df.to_string(index=False))
print(f"\nbest params: weight={best_params['weight']}, C={best_params['C']}, PR-AUC={best_pr_auc:.4f}")
```

     weight    C   pr_auc
         20 1.00 0.996656
         15 1.00 0.996622
         10 1.00 0.996539
          7 1.00 0.996480
          5 1.00 0.996373
         20 0.50 0.996369
         15 0.50 0.996281
          3 1.00 0.996171
         10 0.50 0.996140
          7 0.50 0.995991
          5 0.50 0.995727
          3 0.50 0.995212
          1 1.00 0.994706
         20 0.10 0.993923
         15 0.10 0.993539
         10 0.10 0.992903
          1 0.50 0.992101
          7 0.10 0.992041
         20 0.05 0.991543
          5 0.10 0.990858
         15 0.05 0.990851
         10 0.05 0.989476
          3 0.10 0.987605
          7 0.05 0.987494
          5 0.05 0.984854
          3 0.05 0.981663
         20 0.01 0.979794
         15 0.01 0.978680
          1 0.10 0.978444
         10 0.01 0.976723
          7 0.01 0.974089
          1 0.05 0.970913
          5 0.01 0.970595
          3 0.01 0.963869
          1 0.01 0.950513
    
    best params: weight=20, C=1.0, PR-AUC=0.9967


### Evaluating best model params

Here I train the final model using the best parameters I found during the search. I check how well it performs on both the training and validation sets by calculating the loss, which tells me if the model is learning effectively without overfitting.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Create and train the model with the best parameters found in my search
model = LogisticRegression(
    class_weight={0: 20, 1: 1},  # Higher weight for human class to balance imbalance
    max_iter=5000,
    random_state=42,
    solver="liblinear",
    C=1.0,  # best regularization strength
)

# Fit the model to training data
model.fit(X_train, y_train)
logging.info("model training complete")

# Get probability predictions for both training and validation data
train_probs = model.predict_proba(X_train)
val_probs = model.predict_proba(X_val)

# Calculate loss on both sets to check for overfitting
train_loss = log_loss(y_train, train_probs)
val_loss = log_loss(y_val, val_probs)

# Log and display the loss values
logging.info(f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
print(f"train loss: {train_loss:.4f}")
print(f"val loss: {val_loss:.4f}")
```

    train loss: 0.1995
    val loss: 0.2343


## Testing model

## Testing the Model

To evaluate the final model, I ran it on the held-out test set which consists entirely of `llama-chat` generated text alongside the 20% of human documents that were set aside before training. The reason I chose `llama-chat` as the test model is that it comes from Meta and uses the LLaMA architecture, which is a completely different model family from anything the classifier was trained on. This way, the test is actually measuring whether the model learned general properties of AI writing rather than just memorizing vocabulary patterns from specific models it had already seen.

The results came out to a PR-AUC of 0.9152 on the test set, which is pretty strong given that the model had never seen any LLaMA output during training. Looking at the confusion matrix, the model correctly identified the large majority of both human and AI samples, though it does struggle more with the human class given how underrepresented it is in the dataset.


```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Display detailed performance metrics for each class
print(classification_report(y_test, y_pred, target_names=["human", "AI"]))

# Calculate PR-AUC which is more meaningful for imbalanced datasets
pr_auc = average_precision_score(y_test, y_pred_proba)
print(f"PR-AUC: {pr_auc:.4f}")

# Create confusion matrix to see true positives, false positives, etc.
cm = confusion_matrix(y_test, y_pred)
print(f"confusion matrix:\n{cm}")

logging.info(f"PR-AUC: {pr_auc:.4f}")
```

                  precision    recall  f1-score   support
    
           human       0.48      0.91      0.63       288
              AI       0.99      0.90      0.94      2762
    
        accuracy                           0.90      3050
       macro avg       0.73      0.90      0.78      3050
    weighted avg       0.94      0.90      0.91      3050
    
    PR-AUC: 0.9959
    confusion matrix:
    [[ 262   26]
     [ 284 2478]]


## Visualizations 

Here I create visual charts to better understand my model's performance. I show a confusion matrix that displays how many texts were correctly and incorrectly classified, and a precision-recall curve that shows the trade-off between catching all AI texts and avoiding false alarms.


```python
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set up a figure with two side-by-side charts
fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, figure=fig)

# Create and plot the confusion matrix on the left
ax1 = fig.add_subplot(gs[0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human", "AI"])
disp.plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
ax1.set_xlabel("Predicted Label", fontsize=11)
ax1.set_ylabel("True Label", fontsize=11)

# Create and plot the precision-recall curve on the right
ax2 = fig.add_subplot(gs[1])
PrecisionRecallDisplay.from_predictions(
    y_test, y_pred_proba,
    ax=ax2,
    name=f"Logistic Regression (PR-AUC = {pr_auc:.4f})",
    color="steelblue"
)
ax2.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
ax2.set_xlabel("Recall", fontsize=11)
ax2.set_ylabel("Precision", fontsize=11)
ax2.legend(fontsize=9)

# Add title and save the figure
plt.suptitle("Test Set Evaluation of AI vs Human Text Classifier", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("evaluation.png", dpi=300, bbox_inches="tight")
plt.show()
logging.info("evaluation plots saved to evaluation.png")
```


    
![png](pipeline_files/pipeline_17_0.png)
    


$$Figure \ 1$$

### Visualization Rationale 

- The above confusion matrix and the precision recall curve demonstrates that the model has been optimized to maximally identify all instances (or most) of AI generated text and distinguish them from human generated texts. 
- The **Confusion Matrix** shows that about 262 human texts were correctly identified as human, 2,476 AI texts were correctly identified as AI, 26 Human texts were wrongly marked as AI (False Positives), and 284 AI texts were wrongly marked as Human (False negatives). With a strong class imbalance with many more test cases of AI generated text, I think this shows that the model here is pretty descent at catching AI texts. 
- The **Precision-Recall Curve** shows that our precision recall, which is a metric optimized for class imbalanced datasets, is 0.959, which is near perfect. The curve is staying near the top which shows that my model maintains high precision during testing. 
- Overall, the model missed about 10% of the Ai generated text in the test set, which is a slight set back with my current trained model. 

## Manually Testing

In this final section, I test my trained model on custom text inputs to see how it performs on real examples. I provide a sample of human-written (**My own written language**) and AI-generated (**Claude Opus 4.6**) text and let the model classify them, in which the results show that the model does indeed accurately classify these results


```python
# Prepare sample text inputs to test manually
test_inputs = [
    """Hi Davene!

Sorry for not reaching back to you sooner, I never knew my research was funded by Deloitte! This is also a surprising news to me

I've only been working on the research with Dr.Li in the domain of 3D generative models, such as mechanistically interpreting how 3D generative models work internally, and exploring ways to optimize these models from generating longer context 3D structures.

I'm not too sure as to how the funding maybe have permitted this research that otherwise wouldn't have been possible. I am using the university's HPC GPUs to run the models, so I guess that's where most of the cost may be coming as I am using our lab's allocation to access the GPUs.

My mentor (Dr.Li) is a very knowledgeable and cool person to work with. He knows, and have worked on, a lot of research projects and papers within the 3D spatial scenery reconstruction or generative model domain. I think the lab he was previously at back at TUM was the lab that first employed Transformers to create a GPT like 3D mesh generator, which is really impressive in terms of demonstrating that 3D models can now be generated by an algorithm that was originally designed for text or image generation.

He provided a lot of guidance for me to navigate the space of researching on multimodal generative models. He helped me to decompose complex ideas into simple terms, provided guidance as to what kind of experiments I could work on, novel models/papers I should read about in the industry, as well as potential ideas for the overall research direction. Often times I might get lost in my experiments or just lost about the research direction overall, and I would always turn to him for guidance. We meet weekly in our lab meetings and that's where I'll usually receive feedback about my current progress, as well as giving me directions and inspirations for the future experiments that I could work on.

Let me known if there's anymore info you would like me to provide!
""",
    """The rapid advancement of artificial intelligence technologies has fundamentally transformed the landscape of modern computing and data processing. Machine learning algorithms, particularly deep neural networks, have demonstrated remarkable capabilities in pattern recognition, natural language understanding, and predictive analytics across diverse application domains. These sophisticated systems leverage vast quantities of training data to identify complex statistical relationships that would be virtually impossible for human analysts to detect manually. Furthermore, the integration of transformer architectures has enabled unprecedented performance improvements in sequential data processing tasks, facilitating breakthroughs in automated translation, text summarization, and conversational AI systems. As these technologies continue to mature, researchers are increasingly focused on addressing critical challenges related to model interpretability, computational efficiency, and ethical deployment considerations.""",
    """
    They're actively seeking UVA students to help engineer, acquire, and launch the next generation of space capabilities--join them for a series of events across campus on April 28, 2026 to explore some incredible internship and job opportunities. 1st through 4th years and graduate students are all welcome!
    """
]

# Transform the text inputs using the fitted TF-IDF vectorizer
X_manual_tfidf = tfidf.transform(test_inputs)

# Calculate handcrafted linguistic features for the test inputs
manual_hand = sp.csr_matrix([
    [len(t.split()),
     len(t.split()) / max(len([s for s in t.split('.') if s.strip()]), 1),
     len(set(t.split())) / max(len(t.split()), 1),
     sum(len(w) for w in t.split()) / max(len(t.split()), 1),
     sum(1 for c in t if c in '.,;:!?') / max(len(t), 1),
     0]
    for t in test_inputs
])

# Combine the TF-IDF and handcrafted features
X_manual = sp.hstack([X_manual_tfidf, manual_hand])

# Get class predictions and probability scores
preds = model.predict(X_manual)
probs = model.predict_proba(X_manual)

# Display results for each test input
for i, text in enumerate(test_inputs):
    label = "AI" if preds[i] == 1 else "Human"
    print(f"Text: {text[:80]}...")
    print(f"Prediction: {label} | P(human): {probs[i][0]:.4f}, P(AI): {probs[i][1]:.4f}\n")
```

    Text: Hi Davene!
    
    Sorry for not reaching back to you sooner, I never knew my research ...
    Prediction: Human | P(human): 0.7133, P(AI): 0.2867
    
    Text: The rapid advancement of artificial intelligence technologies has fundamentally ...
    Prediction: AI | P(human): 0.1675, P(AI): 0.8325
    
    Text: 
        They're actively seeking UVA students to help engineer, acquire, and launch...
    Prediction: AI | P(human): 0.1345, P(AI): 0.8655
    



```python
# Interactive prediction loop for custom text input
print("Type any text and press Enter to get a prediction.")
print("Type 'quit' to stop.\n")

while True:
    user_text = input("Enter text: ").strip()

    if user_text.lower() in ["quit", "exit", "q"]:
        print("Stopped interactive predictions.")
        break

    if not user_text:
        print("Please enter some text.\n")
        continue

    # Transform input text with the fitted TF-IDF vectorizer
    X_user_tfidf = tfidf.transform([user_text])

    # Build handcrafted features for the input text
    words = user_text.split()
    sentence_count = max(len([s for s in user_text.split('.') if s.strip()]), 1)
    word_count = len(words)

    user_hand = sp.csr_matrix([[
        word_count,
        word_count / sentence_count,
        len(set(words)) / max(word_count, 1),
        sum(len(w) for w in words) / max(word_count, 1),
        sum(1 for c in user_text if c in '.,;:!?') / max(len(user_text), 1),
        0
    ]])

    # Combine all features and predict
    X_user = sp.hstack([X_user_tfidf, user_hand])
    pred = model.predict(X_user)[0]
    prob = model.predict_proba(X_user)[0]

    label = "AI" if pred == 1 else "Human"
    print(f"Prediction: {label} | P(human): {prob[0]:.4f}, P(AI): {prob[1]:.4f}\n")
```

    Type any text and press Enter to get a prediction.
    Type 'quit' to stop.
    
    Prediction: AI | P(human): 0.4141, P(AI): 0.5859
    
    Prediction: Human | P(human): 0.5345, P(AI): 0.4655
    
    Prediction: AI | P(human): 0.3837, P(AI): 0.6163
    
    Prediction: AI | P(human): 0.1345, P(AI): 0.8655
    
    Please enter some text.
    
    Please enter some text.
    
    Stopped interactive predictions.


## Does this Solves the problem?

- based on the current test results, both from RAID's own test sets and my own custom inputs, I believe that I have indeed found a way to truly allow models (even models as simple as logistic regression) to interpret and distinguish between human and AI generated text. 
- Although due to the class imbalance problem in the dataset, I have to manually add weights to the human class to ensure that the model does not overfit, I think a better way forward is likely to find a better and more balanced dataset so that the model learns sufficient data from both human and AI generated texts. 
- All in all, I would only trust deploying this model once more users have put in their OWN custom texts into this model to test it out pushing it out to the public. But at this moment in time, it does seem like the model that i've built here sufficiently addresses the issue of human and AI texts being indistinguishable amid the spread of AI generated misinformation on social media. 

## Interpreting the coefficients that influenced the model's decisions


```python
# Get the feature names from TF-IDF and handcrafted features
tfidf_feature_names = tfidf.get_feature_names_out()         # 5000 TF-IDF token names
handcrafted_names = np.array(feature_cols)                   # 6 handcrafted feature names
all_feature_names = np.concatenate([tfidf_feature_names, handcrafted_names])  # combine both

# Extract logistic regression coefficients (positive = predicts AI, negative = predicts human)
coefficients = model.coef_[0]

# Get top 20 most positive (AI-predicting) and top 20 most negative (human-predicting) features
top_ai_idx    = np.argsort(coefficients)[-20:][::-1]   # 20 highest coefficients
top_human_idx = np.argsort(coefficients)[:20]          # 20 lowest coefficients

# Combine into one set for plotting
top_idx = np.concatenate([top_ai_idx, top_human_idx])
top_names = all_feature_names[top_idx]
top_coefs = coefficients[top_idx]
top_colors = ["tomato" if c > 0 else "steelblue" for c in top_coefs]  # red = AI, blue = human

# Sort by coefficient value for a cleaner horizontal bar chart
sorted_order = np.argsort(top_coefs)
top_names = top_names[sorted_order]
top_coefs = top_coefs[sorted_order]
top_colors = [top_colors[i] for i in sorted_order]

fig, ax = plt.subplots(figsize=(10, 12))
bars = ax.barh(top_names, top_coefs, color=top_colors)   
ax.axvline(x=0, color="black", linewidth=0.8)           
ax.set_xlabel("Logistic Regression Coefficient", fontsize=12)
ax.set_title("Top 20 Features Predicting AI vs Human Text\n(Red = AI, Blue = Human)", 
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()
logging.info("feature importance plot saved to feature_importance.png")
```


    
![png](pipeline_files/pipeline_28_0.png)
    


- This plot shows the words that are learned by the model to be used to distinguish AI and human language, it is rather interesting to see how AI generated language has a lot of full words while human language usually has simpler filler words


```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

features_to_plot = ["word_count", "unique_word_ratio", "avg_word_length"]
titles = ["Word Count", "Unique Word Ratio", "Avg Word Length"]
clips = [1000, None, 10]  # clip outliers per feature

for ax, feat, title, clip in zip(axes, features_to_plot, titles, clips):
    human_vals = df[df["label"] == 0][feat]
    ai_vals    = df[df["label"] == 1][feat]

    if clip:
        human_vals = human_vals.clip(upper=clip)
        ai_vals    = ai_vals.clip(upper=clip)

    parts_human = ax.violinplot([human_vals], positions=[0], showmedians=True)
    parts_ai    = ax.violinplot([ai_vals],    positions=[1], showmedians=True)

    # Color human steelblue, AI tomato
    for pc in parts_human["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.7)
    for pc in parts_ai["bodies"]:
        pc.set_facecolor("tomato")
        pc.set_alpha(0.7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Human", "AI"], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Value", fontsize=10)

plt.suptitle("Linguistic Feature Distributions: Human vs AI Text",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("feature_distributions.png", dpi=300, bbox_inches="tight")
plt.show()
logging.info("feature distribution plot saved to feature_distributions.png")
```


    
![png](pipeline_files/pipeline_30_0.png)
    


- The differences between the statistical differences in human and AI generated language, AI seem to have a narrower range of word count, and wide range of unique words, and more outliers in terms of average word length. Whereas human writings are more stochastic on word count (the word count varies a lot), the unique words employed is also pretty fixed, and the average word length is pretty fixed too.
