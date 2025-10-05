import heapq
import string
import itertools

# 1. Text Preprocessing
def preprocess(text):
    sentences = text.split('.')
    sentences = [s.strip().lower().translate(str.maketrans('', '', string.punctuation))
                 for s in sentences if s.strip()]
    return sentences

# 2. Levenshtein Distance
def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# 3. A* Search for Alignment
def astar_search(doc1_sents, doc2_sents):
    m, n = len(doc1_sents), len(doc2_sents)
    counter = itertools.count()

    def heuristic(i, j):
        rem1 = m - i
        rem2 = n - j
        if rem1 > rem2:
            lens = sorted(len(s) for s in doc1_sents[i:])
            return sum(lens[:rem1 - rem2])
        elif rem2 > rem1:
            lens = sorted(len(s) for s in doc2_sents[j:])
            return sum(lens[:rem2 - rem1])
        else:
            return 0

    start = (0, 0)
    goal = (m, n)
    heap = [(0 + heuristic(0, 0), next(counter), 0, 0, 0, [])]  # (f, count, g, i, j, path)
    visited = set()

    while heap:
        f, _, g, i, j, path = heapq.heappop(heap)
        if (i, j) == goal:
            return path
        if (i, j) in visited:
            continue
        visited.add((i, j))

        # Align sentences
        if i < m and j < n:
            dist = levenshtein(doc1_sents[i], doc2_sents[j])
            new_g = g + dist
            new_f = new_g + heuristic(i + 1, j + 1)
            heapq.heappush(heap, (new_f, next(counter), new_g, i + 1, j + 1,
                                  path + [((i, j), dist)]))

        # Skip sentence in doc1
        if i < m:
            penalty = len(doc1_sents[i])
            new_g = g + penalty
            new_f = new_g + heuristic(i + 1, j)
            heapq.heappush(heap, (new_f, next(counter), new_g, i + 1, j,
                                  path + [((i, None), penalty)]))

        # Skip sentence in doc2
        if j < n:
            penalty = len(doc2_sents[j])
            new_g = g + penalty
            new_f = new_g + heuristic(i, j + 1)
            heapq.heappush(heap, (new_f, next(counter), new_g, i, j + 1,
                                  path + [((None, j), penalty)]))

    return []

# 4. Detect Plagiarism
def detect_plagiarism(alignment_path, doc1_sents, doc2_sents, threshold=40):
    potential = []
    for (i, j), dist in alignment_path:
        if i is not None and j is not None:   # Skip invalid pairs
            if dist <= threshold:
                potential.append((doc1_sents[i], doc2_sents[j], dist))
    return potential

# 5. Evaluate a Pair
def evaluate_pair(docA, docB, label, threshold=40):
    docA_sents = preprocess(docA)
    docB_sents = preprocess(docB)
    alignment_path = astar_search(docA_sents, docB_sents)

    print(f"\n===== Results for {label} =====")
    print("Doc A sentences:", docA_sents)
    print("Doc B sentences:", docB_sents)

    plagiarism_pairs = detect_plagiarism(alignment_path, docA_sents, docB_sents, threshold)

    if plagiarism_pairs:
        print("\nPotential Plagiarism Detected:")
        for s1, s2, dist in plagiarism_pairs:
            print(f' - "{s1}"  <-->  "{s2}"  (distance={dist})')
    else:
        print("\nNo plagiarism detected.")

# 6. Main
def main():
    # Test Case 1: Identical Documents
    doc1 = """Artificial intelligence is the study of agents that perceive their environment and take actions.
              Plagiarism detection is important in academia.
              Text alignment is one way to find similar sentences."""
    doc2 = """Artificial intelligence is the study of agents that perceive their environment and take actions.
              Plagiarism detection is important in academia.
              Text alignment is one way to find similar sentences."""

    # Test Case 2: Slightly Modified Document
    doc3 = """Machine learning is a subset of artificial intelligence.
              It enables systems to learn and improve from experience.
              Data preprocessing is crucial."""
    doc4 = """Machine learning comes under artificial intelligence.
              It allows computers to learn from data.
              Preprocessing data is an important step."""

    # Test Case 3: Completely Different Documents
    doc5 = """The capital of France is Paris.
              The Eiffel Tower is one of the most famous monuments.
              French cuisine includes dishes like baguettes and croissants."""
    doc6 = """The solar system consists of the sun and the planets.
              Earth revolves around the sun.
              Space exploration has advanced significantly in recent decades."""

    # Test Case 4: Partial Overlap
    doc7 = """Artificial intelligence is the study of agents that perceive their environment and take actions.
              Machine learning is a subset of artificial intelligence.
              The capital of France is Paris."""
    doc8 = """Artificial intelligence is the study of agents that perceive their environment and take actions.
              Plagiarism detection is important in academia.
              Machine learning is a subset of artificial intelligence."""

    evaluate_pair(doc1, doc2, "Test Case 1: Identical Documents")
    evaluate_pair(doc3, doc4, "Test Case 2: Slightly Modified Document")
    evaluate_pair(doc5, doc6, "Test Case 3: Completely Different Documents")
    evaluate_pair(doc7, doc8, "Test Case 4: Partial Overlap")

if _name_ == "_main_":
    main()
