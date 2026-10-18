from ingest import load_pdfs, chunk_pages
from retrieve import search

# (question, part of the PDF name, keyword that must be in the correct chunk, all lowercase)
TESTS = [
    # --- cross-validation ---
    ("How do I split data into training and test sets?", "cross-validation", "train_test_split"),
    ("Why can knowledge about the test set leak into the model?", "cross-validation", "leak"),
    ("What is k-fold cross-validation?", "cross-validation", "fold"),
    ("Why shouldn't I test a model on the same data it learned from?", "cross-validation", "overfitting"),
    ("How do I get cross-validation scores with one function call?", "cross-validation", "cross_val_score"),
    ("Which splitter keeps the class proportions the same in each fold?", "cross-validation", "stratified"),
    ("What is leave-one-out cross-validation?", "cross-validation", "leaveoneout"),
    ("How do I cross-validate data that is ordered in time?", "cross-validation", "timeseriessplit"),
    # --- gaussian processes ---
    ("What is a Gaussian Process?", "gaussian", "nonparametric"),
    ("What is a good thing about Gaussian processes when making predictions?", "gaussian", "interpolates"),
    ("When do Gaussian processes stop working well?", "gaussian", "dozens"),
    ("What defines the prior in a Gaussian process?", "gaussian", "covariance"),
    ("Which class do I use for classification with Gaussian processes?", "gaussian", "gaussianprocessclassifier"),
    # --- pls-canonical ---
    ("How many components does PLSCanonical keep by default?", "pls-canonical", "n_components"),
    ("Which algorithm can PLSCanonical use to compute the weights?", "pls-canonical", "nipals"),
    ("Does PLSCanonical scale the data before fitting?", "pls-canonical", "scale"),
    ("How do I convert transformed data back to the original space?", "pls-canonical", "inverse_transform"),
    # --- visualizations ---
    ("How do I plot a confusion matrix?", "visualizations", "confusionmatrixdisplay"),
    ("How can I draw an ROC curve?", "visualizations", "roccurvedisplay"),
    ("How do I create a plot object directly from a fitted model?", "visualizations", "from_estimator"),
    ("Can I put several plots on the same figure?", "visualizations", "axes"),
    # --- choosing-estimator ---
    ("How should I read the orange arrows in the estimator chart?", "choosing-estimator", "arrows"),
    ("What is the hardest part of solving a machine learning problem?", "choosing-estimator", "hardest"),
    ("What is the estimator flowchart meant to give users?", "choosing-estimator", "rough guide"),
]

all_chunks = chunk_pages(load_pdfs())


def keyword_exists(source, keyword):
    return any(source in c["source"].lower() and keyword in c["text"].lower()
               for c in all_chunks)


hit5 = hit1 = used = 0
for question, source, keyword in TESTS:
    if not keyword_exists(source, keyword):
        print(f"SKIP  keyword '{keyword}' not found in {source}: {question}")
        continue
    used += 1
    results = search(question, k=5)
    flags = [source in meta["source"].lower() and keyword in text.lower()
             for text, meta in results]
    hit5 += any(flags)
    hit1 += flags[0]
    print(("HIT " if any(flags) else "MISS") + ("  top1 " if flags[0] else "       ") + question)

if used:
    print(f"\nQuestions used: {used} of {len(TESTS)}")
    print(f"Top-5 hit rate: {hit5}/{used} = {hit5 / used:.0%}")
    print(f"Top-1 hit rate: {hit1}/{used} = {hit1 / used:.0%}")