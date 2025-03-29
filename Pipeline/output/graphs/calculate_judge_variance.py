import re
import numpy as np
import matplotlib.pyplot as plt

# Load the file content
file_path = "europa_judge_verdicts.txt"
with open(file_path, "r", encoding="utf-8") as file:
    content = file.read()

# Extract all language sections and their scores
pattern = r"==============Language: (\w+)=+[\s\S]*?Parsed Numeric Scores:\s*([\d\.,\s]+)"
matches = re.findall(pattern, content)

# Store variances and means for plotting or inspection
language_variances = {}
language_means = {}

for lang, score_str in matches:
    # Parse the numeric scores
    scores = [float(s.strip()) for s in score_str.strip().split(",") if s.strip()]
    variance = np.var(scores)
    mean_score = np.mean(scores)
    language_variances[lang] = variance
    language_means[lang] = mean_score

# Plotting
languages = list(language_variances.keys())
variances = [language_variances[lang] for lang in languages]

plt.figure(figsize=(10, 6))
plt.bar(languages, variances)
plt.title("Variance of Judged Scores by Language")
plt.xlabel("Language")
plt.ylabel("Score Variance")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
