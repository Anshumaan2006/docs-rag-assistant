from retrieve import search

# (question, part of the correct PDF's name in lowercase, keyword in the correct chunk in lowercase)
TESTS = [
    ("How do I split data into training and test sets?", "cross-validation", "train_test_split"),
    ("Why can knowledge about the test set leak into the model?", "cross-validation", "leak"),
    ("What is a Gaussian Process?", "gaussian", "nonparametric"),
    ("What is k-fold cross-validation?", "cross-validation", "fold"),
    # add at least 8 more, from choosing-estimator, visualizations, and pls-canonical too
]

hit5 = hit1 = 0
for question, source, keyword in TESTS:
    results = search(question, k=5)
    flags = [source in meta["source"].lower() and keyword in text.lower()
             for text, meta in results]
    hit5 += any(flags)
    hit1 += flags[0]
    print(("HIT " if any(flags) else "MISS") + ("  top1 " if flags[0] else "       ") + question)

n = len(TESTS)
print(f"\nTop-5 hit rate: {hit5}/{n} = {hit5 / n:.0%}")
print(f"Top-1 hit rate: {hit1}/{n} = {hit1 / n:.0%}")