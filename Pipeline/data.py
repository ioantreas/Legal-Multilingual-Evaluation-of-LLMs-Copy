import copy
import json

import os
import csv

import unicodedata
from datasets import load_dataset, concatenate_datasets
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, average_precision_score
from sklearn.preprocessing import MultiLabelBinarizer
from translator import translate
import numpy as np
import textwrap

import re
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
from collections import Counter
from deep_translator import GoogleTranslator
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rapidfuzz import fuzz

from utils import get_embedding_bert, get_language_from_code, store_judge
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import precision_score, recall_score, f1_score



class Dataset:
    """
    Base Dataset class with a.py factory method to return the appropriate dataset object.
    """

    def get_data(self, language, dataset_name, points_per_language):
        """
        Abstract method to get data in a.py specific language.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    def get_true(self, data):
        """
        Abstract method to get the true labels/text for the dataset.
        This should be implemented by child classes.
        """
        raise NotImplementedError("Child class must implement this method")

    @staticmethod
    def get_dataset(name, llm_judge):
        """
        :param name: name of the dataset
        :return: the dataset object
        """
        if name.lower() == 'multi_eurlex':
            return Multi_Eurlex(llm_judge)
        elif name.lower() == 'eur_lex_sum':
            return Eur_Lex_Sum()
        elif name.lower() == 'europa_random_split':
            return Europa_Random_Split(llm_judge)
        elif name.lower() == 'covid19':
            return Covid19EmergencyEvent()
        elif name.lower() == 'terms_of_service':
            return OnlineTermsOfServiceDataset()
        elif name.lower() == 'casehold':
            return CaseHOLD()
        elif name.lower() == 'xquad':
            return XQuAD(llm_judge)
        elif name.lower() == 'xnli':
            return XNLI()
        elif name.lower() == 'go_emotions':
            return Go_Emotions()
        elif name.lower() == 'sst2':
            return SST2()
        elif name.lower() == 'qqp':
            return QQP()
        elif name.lower() == 'mnli':
            return MNLI()
        elif name.lower() == 'qnli':
            return QNLI()
        else:
            raise ValueError(f"Dataset '{name}' is not available")

    def normalize_text(self, text):
        # Convert to lowercase and remove accents
        text = text.lower()
        return ''.join(
            c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'
        )

    def extract_labels_from_generated_text(self, generated_text, label_options):
        cleaned_text = self.normalize_text(generated_text.replace("\u200B", ""))
        relevant_labels = []
        for i, label in enumerate(label_options):
            cleaned_label = self.normalize_text(label.replace("\u200B", ""))
            # Use \b to ensure the label is a.py standalone word or phrase
            pattern = r'\b' + re.escape(cleaned_label) + r'\b'
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                relevant_labels.append(i)
        return relevant_labels


    ###########################################################################################
    ###################################### Legal Datasets #####################################
    ###########################################################################################

class Multi_Eurlex(Dataset):
    """
    Child class of Dataset representing the Multi-EUR-Lex dataset.
    """

    label_options = None

    def __init__(self, llm_judge):
        self.prompt = (
            "<|endoftext|>\n\n\nYou are a legal document classifier. Above is a legal document and below is a list of possible labels.\n"
            "Your task is to assign the most relevant labels based on the content of the document.\n"
            "- You may select multiple labels.\n"
            "- Only select relevant ones.\n"
            "Return **only the label numbers**, separated by commas, in order of importance (most important first).\n"
            "- Do not explain your answer or include any other text.\n\n"
            "Label Options:\n"
        )

        self.llm_judge = llm_judge

    def load_label_options(self, lang):
        # Load files
        with open("data/multi_eurlex/eurovoc_concepts.json", "r", encoding="utf-8") as f:
            concepts = json.load(f)

        with open("data/multi_eurlex/eurovoc_descriptors.json", "r", encoding="utf-8") as f:
            descriptors = json.load(f)

        # Get Level 3 IDs
        level_3_ids = concepts["level_3"]

        # Filter Level 3 descriptors with IDs
        level_3_label_tuples = []
        for concept_id in level_3_ids:
            label = descriptors.get(concept_id, {}).get(lang)
            if label:
                level_3_label_tuples.append((concept_id, label.strip().lower()))

        # Build concept_id → index mapping (starting from 0)
        self.concept_id_to_index = {
            cid: i for i, (cid, _) in enumerate(level_3_label_tuples)
        }

        # Return just the list of label strings (not numbered)
        return [label for _, label in level_3_label_tuples]


    def load_level_3_ids(self):
        with open("data/multi_eurlex/eurovoc_concepts.json", "r", encoding="utf-8") as f:
            concepts = json.load(f)
        return set(concepts["level_3"])

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """
        self.lang = language
        dataset = load_dataset('coastalcph/multi_eurlex', language, split='test', label_level='level_3', trust_remote_code=True)

        # Get the mapping from label indices to concept IDs
        self.label_id_to_concept = dataset.features["labels"].feature.names

        # Load level 3 IDs for filtering
        self.level_3_ids = self.load_level_3_ids()

        # Load label options in the target language, only for Level 3
        self.label_options = self.load_label_options(language)

        # Get the processed document-text + labels
        data = self.extract_text(dataset)

        # Translate prompt
        inst = translate(language, self.prompt)
        return data[:points_per_language], self.label_options, inst

    def extract_text(self, dataset):
        preprocessed_data = []

        for item in dataset:
            text = item['text']
            label_indices = [i for i in item['labels'] if 0 <= i < len(self.label_options)]
            preprocessed_data.append({
                "text": text,
                "labels": label_indices
            })

        return preprocessed_data

    def get_true(self, data):
        """
        :return: a.py list of true labels for the dataset
        """
        true_labels = [entry['labels'] for entry in data]
        return true_labels

    def extract_labels_from_generated_text(self, generated_texts):
        """
        :param generated_texts: list of generated texts
        :return: a list of predicted labels for each generated text, in order of appearance
        """
        if self.llm_judge:
            return [text for text in generated_texts]

        all_labels = []
        for text in generated_texts:
            if not isinstance(text, str):
                all_labels.append([])
                continue

            # Build a list of (position, label) tuples
            matches = []
            for i in range(len(self.label_options)):
                # Match whole word using word boundaries
                for match in re.finditer(rf'\b{i}\b', text):
                    matches.append((match.start(), i))

            # Sort by position and extract labels in order
            ordered_labels = [label for _, label in sorted(matches)]
            all_labels.append(ordered_labels)

        return all_labels


    def extract_label_indices(self, response: str) -> list[int] | None:
        if not response or not isinstance(response, str):
            return None

        response = response.strip().lower()

        if 'none' in response:
            return None

        # Extract all numbers from the response
        numbers = re.findall(r'\d+', response)
        return [int(num) for num in numbers if 0 <= int(num) < len(self.label_options)]


    def evaluate(self, true_labels, predicted_labels):
        if self.llm_judge:
            prompts = []
            for pred_answer in predicted_labels:
                pred_answer = pred_answer or ""
                prompt = (
                        "You are evaluating how well a model assigned labels to a text. "
                        "Multiple labels may apply. The model may have responded using the label numbers, label names, or both. "
                        f"All content (model answer and labels) is in {get_language_from_code(self.lang)}.\n\n"
                        "Your task:\n"
                        "- Use the numbered 'Labels' list below to determine which labels were identified in the model's answer.\n"
                        "- Return only the numbers of the labels that the model has identified in the order that they were identified, separated by commas.\n"
                        "- If no labels were identified, return 'None'.\n\n"
                        f"Answer: {pred_answer.strip()}\n"
                        "Labels:\n" +
                        "\n".join(f"{i}: {label}" for i, label in enumerate(self.label_options)) + "\n\n"
                                                                                                   "Labels identified:"
                )
                prompts.append(prompt)

            # Batch call to judge
            responses = self.llm_judge.judge(prompts)

            # Extract labels from responses
            labels = [self.extract_label_indices(resp) or [] for resp in responses]
            predicted_labels = labels

            store_judge(responses, labels, self.lang)

        # Ensure predicted_labels has no None values
        predicted_labels = [lbl if lbl is not None else [] for lbl in predicted_labels]

        mlb = MultiLabelBinarizer(classes=list(range(len(self.label_options))))

        # Binarize the true and predicted labels
        binary_true = mlb.fit_transform(true_labels)
        binary_pred = mlb.transform(predicted_labels)

        # Get indices of labels with non-zero true or predicted samples
        relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]

        # Filter binary_true and binary_pred to only include relevant labels
        filtered_binary_true = binary_true[:, relevant_labels]
        filtered_binary_pred = binary_pred[:, relevant_labels]

        # Calculate precision, recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
        )

        # Compute mean R-Precision (mRP)
        r_precisions = []
        for true, pred in zip(true_labels, predicted_labels):
            if not true:
                continue  # Skip samples with no gold labels
            k = len(true)
            top_k_pred = pred[:k]  # Take top-k predicted labels
            correct = len(set(top_k_pred) & set(true))
            r_precision = correct / k
            r_precisions.append(r_precision)

        mean_r_precision = np.mean(r_precisions) if r_precisions else 0.0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "mRP": mean_r_precision,
            "Length": len(true_labels)
        }

    def evaluate_results(self, results):
        # Print out the results for each language
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            print(f"Precision: {metrics['Precision']}")
            print(f"Recall: {metrics['Recall']}")
            print(f"F1 Score: {metrics['F1 Score']}")
            print(f"mRP: {metrics['mRP']}")
            print(f"Length: {metrics['Length']}\n")

    def save_first_10_results_to_file_by_language(self, first_ten_answers, language):
        # Define the output folder path
        output_folder = "output/multi_eurlex/10_first"

        # Create the directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Create a.py filename specific to the language within the output folder
        filename = os.path.join(output_folder, f"gemini_results_{language}.txt")

        # Check if the file exists; if not, create it and write headers
        if not os.path.exists(filename):
            with open(filename, 'w', encoding='utf-8') as file:
                file.write("")

        with open(filename, 'a', encoding='utf-8') as file:
            for i in range(min(10, len(first_ten_answers))):
                text = first_ten_answers[i]
                file.write(f"{text}\n")


class Eur_Lex_Sum(Dataset):
    """
    Child class of Dataset representing the Eur-Lex-sum dataset.
    """

    def __init__(self):
        self.prompt = "\n<|endoftext|>\nTask: Summarize the text above. Include all the important information."
        # self.prompt = "\n<|endoftext|>\nTask: Write something about the text above."

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """
        print("Reached get_data")
        dataset = load_dataset('dennlinger/eur-lex-sum', language, streaming=True, split='train', trust_remote_code=True)
        self.language = language
        data = self.extract_text(dataset, points_per_language)
        inst = translate(language, self.prompt)
        return data, inst

    def extract_text(self, dataset, points_per_language):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == points_per_language:
                break
            data.append({"text": item['reference'], "summary": item['summary']})
            count += 1
        return data

    def get_true(self, data):
        """
        :return: the true summary of the data
        """
        summary = [entry['summary'] for entry in data]
        return summary

    def format_text_to_width(self, text, width):
        """
        Splits a text into lines of a given width.
        """
        return "<br>".join(textwrap.wrap(text, width))

    def evaluate(self, references, predictions):
        rouge = evaluate.load("rouge", cache_dir=f"/tmp/huggingface_cache/{os.getpid()}")

        results_rouge = rouge.compute(predictions=predictions, references=references)
        embedded_references = [get_embedding_bert(reference) for reference in references]
        embedded_predictions = [get_embedding_bert(prediction) for prediction in predictions]
        cosine_similarities = [cosine_similarity(embedded_reference, embedded_prediction) for (embedded_reference, embedded_prediction) in zip(embedded_references, embedded_predictions)]
        avg_cosine_similarity = np.mean(cosine_similarities)
        results_cosine = {"cosine_similarity": avg_cosine_similarity}
        results = results_rouge | results_cosine

        # Store the first 3 reference and predicted for checking
        file_path = "output/Eur_Lex_Sum_evaluation.md"
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', encoding='utf-8') as f:
            if not file_exists:
                f.write("| Language | Reference Summary                          | Predicted Summary                           |\n")
                f.write("|----------|------------------------------------------|--------------------------------------------|\n")
            count = 0
            for reference, prediction in zip(references, predictions):
                # Wrap text to fit within 50 characters
                formatted_reference = self.format_text_to_width(reference, 50)
                formatted_prediction = self.format_text_to_width(prediction, 50)
                # Write formatted text into md table
                f.write(f"| {self.language} | {formatted_reference} | {formatted_prediction} |\n")
                count += 1
                if count == 3:
                    break

        return results

    def evaluate_results(self, results):
        # Print out the results for each language
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            print(f"Rouge1: {metrics['rouge1']}")
            print(f"Rouge2: {metrics['rouge2']}")
            print(f"RougeL: {metrics['rougeL']}")
            print(f"RougeL sum: {metrics['rougeLsum']}")
            print(f"Cosine Similarity: {metrics['cosine_similarity']}")
            print("-------------------------------------------------------------")


    def extract_labels_from_generated_text(self, generated_text, label_options):
        """
        :param generated_text: the generated text
        :param label_options: the list of label options
        :return: a list of predicted labels for the generated text
        """
        labels = []
        for i in range(len(label_options)):
            # Use regex to match only whole words for each index, avoiding partial matches
            if re.search(rf'\b{i}\b', generated_text):
                labels.append(i)

        return labels


class Europa_Random_Split(Dataset):
    """
    Child class of Dataset representing the Eur-Lex-sum dataset.
    """

    def __init__(self, llm_judge):
        self.prompt = (
            "\n<|endoftext|>\n"
            "Task: Given the text above, extract a list of keyphrases (short phrases that describe the text) that best summarize the content.\n"
            "List the keyphrases one per line, in order of importance (most important first).\n"
            "Include all essential information from the text.\n"
            "Only return the keyphrases—no explanations or additional text."
        )
        self.llm_judge = llm_judge


    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: the language for which data should be retrieved
        :return: the data corresponding to the language parameter
        """

        dataset = load_dataset('NCube/europa-random-split', streaming=True, split='train', trust_remote_code=True)
        filtered_dataset = (example for example in dataset if example["lang"] == language)
        self.language = language
        data = self.extract_text(filtered_dataset, points_per_language)
        inst = translate(language, self.prompt)
        return data, inst

    def extract_text(self, dataset, points_per_language):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == points_per_language:
                break
            data.append({"text": item['input_text'], "keyphrases": item['keyphrases']})
            count += 1
        return data

    def get_true(self, data):
        """
        :return: the true summary of the data
        """
        summary = [entry['keyphrases'] for entry in data]
        return summary

    def format_text_to_width(self, text, width):
        """
        Splits a text into lines of a given width.
        """
        return "<br>".join(textwrap.wrap(text, width))

    def calculate_f1(self, true_set, pred_list, k=None, threshold=70):
        """
        Calculate F1 score using fuzzy matching to account for order insensitivity.
        :param true_set: Set of true keyphrases.
        :param pred_list: List of predicted keyphrases.
        :param k: If specified, use only the top-k predictions.
        :param threshold: Fuzzy matching similarity threshold (0-100).
        :return: Precision, Recall, and F1 score.
        """
        if k:
            pred_list = pred_list[:k]

        matched_true = set()
        matched_pred = set()

        # Iterate over predicted keyphrases
        for pred in pred_list:
            # Find the best match in the true set
            for true in true_set:
                if true not in matched_true and fuzz.ratio(pred, true) >= threshold:
                    matched_true.add(true)
                    matched_pred.add(pred)
                    break

        # Calculate true positives
        true_positives = len(matched_true)

        # Calculate precision and recall
        precision = true_positives / len(pred_list) if len(pred_list) > 0 else 0.0
        recall = true_positives / len(true_set) if len(true_set) > 0 else 0.0

        # Calculate F1 score
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def calculate_map(self, true_set, pred_list, k=50, threshold=70):
        """
        Calculate Mean Average Precision (MAP) at k using fuzzy matching.
        :param true_set: Set of true keyphrases.
        :param pred_list: List of predicted keyphrases.
        :param k: Use only top-k predictions.
        :param threshold: Fuzzy matching similarity threshold (0-100).
        :return: MAP@k score.
        """
        if k:
            pred_list = pred_list[:k]

        matched_true = set()
        binary_relevance = []

        # Iterate over predictions to calculate binary relevance
        for pred in pred_list:
            match_found = False
            for true in true_set:
                if true not in matched_true and fuzz.ratio(pred, true) >= threshold:
                    matched_true.add(true)
                    match_found = True
                    break
            binary_relevance.append(1 if match_found else 0)

        if not binary_relevance:
            return 0.0

        # Calculate precision at each relevant index (1-based)
        relevant_indices = [i + 1 for i, rel in enumerate(binary_relevance) if rel == 1]
        precisions = [sum(binary_relevance[:i]) / i for i in relevant_indices]

        # Return mean average precision
        return sum(precisions) / len(relevant_indices) if relevant_indices else 0.0

    def evaluate(self, references, predictions):
        """
        Evaluate predictions using either LLM-based semantic judgment or classic F1/MAP metrics.

        :param references: List of lists of ground truth keyphrases.
        :param predictions: List of strings (predicted keyphrases, separated by newlines).
        :param texts: List of original input texts (required for LLM judge).
        :return: Dictionary of evaluation metrics.
        """
        if self.llm_judge:
            prompts = []
            for true_list, pred_str in zip(references, predictions):
                true_keyphrases = "\n".join(true_list)
                pred_keyphrases = pred_str.strip()

                prompt = (
                    "You are evaluating how well a generated list of keyphrases produced by a model summarizes a given text.\n"
                    f"The content is in {get_language_from_code(self.language)}. Use the real (reference) keyphrases as a gold standard.\n\n"
                    "Your task is to rate how well the generated keyphrases capture the meaning and key ideas of the text, using the following scale:\n\n"
                    "5 - Excellent: The keyphrases cover all essential topics and match the reference very closely in meaning.\n"
                    "4 - Good: Most important topics are covered with only minor omissions or differences.\n"
                    "3 - Fair: Some important information is missing, or keyphrases are partially incorrect.\n"
                    "2 - Poor: Few key ideas are captured correctly; many important ones are missing.\n"
                    "1 - Very poor: The keyphrases do not reflect the text meaningfully.\n\n"
                    "Return only the number.\n\n"
                    f"Reference keyphrases:\n{true_keyphrases.strip()}\n\n"
                    f"Generated keyphrases (answer produced by a model):\n{pred_keyphrases}\n\n"
                    "Score:"
                )

                prompts.append(prompt)

            scores = self.llm_judge.judge(prompts)

            # Convert scores to numeric
            numeric_scores = []
            for raw_score in scores:
                if isinstance(raw_score, (int, float)):
                    numeric_scores.append(float(raw_score))
                    continue

                if not isinstance(raw_score, str):
                    numeric_scores.append(0.0)
                    continue

                match = re.search(r"\b([1-5](?:\.0)?)\b", raw_score)
                if match:
                    try:
                        numeric_scores.append(float(match.group(1)))
                    except ValueError:
                        numeric_scores.append(0.0)
                else:
                    numeric_scores.append(0.0)

            store_judge(scores, numeric_scores, self.language)

            #     return {
            #         "LLM Similarity Score (1–5)": np.mean(numeric_scores) if numeric_scores else 0.0
            #     }
            # else:
            metrics = {
                "F1@5": [],
                "F1@10": [],
                "F1@M": [],
                "MAP@50": []
            }

            for ref, pred_str in zip(references, predictions):
                # Convert predictions to a list of keyphrases
                pred_list = [phrase.strip() for phrase in pred_str.split('\n') if phrase.strip()]

                # Convert references to a set for comparison
                true_set = set(ref)

                # Calculate F1@5, F1@10, and F1@M
                _, _, f1_5 = self.calculate_f1(true_set, pred_list, k=5)
                _, _, f1_10 = self.calculate_f1(true_set, pred_list, k=10)
                _, _, f1_m = self.calculate_f1(true_set, pred_list)

                # Calculate MAP@50
                map_50 = self.calculate_map(true_set, pred_list, k=50)

                # Append metrics for this instance
                metrics["F1@5"].append(f1_5)
                metrics["F1@10"].append(f1_10)
                metrics["F1@M"].append(f1_m)
                metrics["MAP@50"].append(map_50)

            # Aggregate metrics across all instances
            aggregated_metrics = {metric: sum(scores) / len(scores) if scores else 0.0 for metric, scores in metrics.items()}

            aggregated_metrics["LLM Similarity Score"] = np.mean(numeric_scores) if numeric_scores else 0.0

            return aggregated_metrics

    def evaluate_results(self, results):
        """
        Display aggregated F1@k, F1@M, and MAP@50 results.

        :param results: Dictionary where each key is a language (e.g., en, el)
                        and the value is a map of metrics and scores.
        """
        for language, scores in results.items():
            print(f"{language}:")
            for metric, score in scores.items():
                print(f"  {metric}: {score:.3f}")

class Covid19EmergencyEvent(Dataset):
    """
    Child class of Dataset representing the COVID-19 Emergency Event dataset.
    """

    def __init__(self):
        self.label_options = None
        self.prompt = (
            "<|endoftext|>\n\n\nYou are a legal document classifier. Above is a legal document and below is a list of possible measures types.\n"
            "Your task is to assign the most relevant types based on the content of the document.\n"
            "- You may select multiple labels.\n"
            "- Only select relevant ones.\n"
            "Return **only the label numbers**, separated by commas, in order of importance (most important first).\n"
            "- Do not explain your answer or include any other text.\n\n"
            "Label Options:\n"
        )

    def get_data(self, language, dataset_name, points_per_language):
        """
        :param language: ISO code (e.g., 'fr', 'en')
        :param dataset_name: unused
        :param points_per_language: how many points to return
        :return: (data, label_options, prompt)
        """
        # Load dataset and filter by language
        dataset_dict = load_dataset("joelniklaus/covid19_emergency_event")

        # Combine train + validation + test into one unified dataset
        dataset = concatenate_datasets([
            dataset_dict['train'],
            dataset_dict['validation'],
            dataset_dict['test']
        ])
        dataset = dataset.filter(lambda x: x["language"] == language and len(x["all_events"]) > 0)

        # Load label translations from a single file, then select the language
        with open("data/covid19_emergency_event/covid19_measures.json", "r", encoding="utf-8") as f:
            all_labels = json.load(f)
            self.label_options = all_labels[language]

        # Preprocess and extract texts and label indices
        data = self.extract_text(dataset)

        inst = translate(language, self.prompt)
        return data[:points_per_language], self.label_options, inst

    def extract_text(self, dataset):
        preprocessed_data = []

        for item in dataset:
            text = item["text"]
            events = item.get("all_events", [])

            # Convert event names like "event3" to indices (i.e., 2)
            label_indices = [
                int(re.sub(r"event", "", event)) - 1  # start from 0
                for event in events
                if event.startswith("event")
            ]

            preprocessed_data.append({
                "text": text,
                "labels": label_indices
            })

        return preprocessed_data

    def get_true(self, data):
        return [entry["labels"] for entry in data]

    def extract_labels_from_generated_text(self, generated_texts):
        all_labels = []
        for text in generated_texts:
            if not isinstance(text, str):
                all_labels.append([])
                continue

            matches = []
            for i in range(len(self.label_options)):
                for match in re.finditer(rf"\b{i}\b", text):
                    matches.append((match.start(), i))

            ordered_labels = [label for _, label in sorted(matches)]
            all_labels.append(ordered_labels)
        return all_labels

    def extract_label_indices(self, response: str) -> list[int] | None:
        if not response or not isinstance(response, str):
            return None
        response = response.strip().lower()
        if "none" in response:
            return None
        numbers = re.findall(r"\d+", response)
        return [int(num) for num in numbers if 0 <= int(num) < len(self.label_options)]

    def evaluate(self, true_labels, predicted_labels):
        predicted_labels = [lbl if lbl is not None else [] for lbl in predicted_labels]
        mlb = MultiLabelBinarizer(classes=list(range(len(self.label_options))))
        binary_true = mlb.fit_transform(true_labels)
        binary_pred = mlb.transform(predicted_labels)

        # Filter to relevant labels (used at least once)
        relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]
        filtered_binary_true = binary_true[:, relevant_labels]
        filtered_binary_pred = binary_pred[:, relevant_labels]

        # Per-sample F1
        per_sample_f1 = []
        for t, p in zip(filtered_binary_true, filtered_binary_pred):
            precision_i = np.sum(t & p) / np.sum(p) if np.sum(p) > 0 else 0.0
            recall_i = np.sum(t & p) / np.sum(t) if np.sum(t) > 0 else 0.0
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0.0
            per_sample_f1.append(f1_i)

        mean_f1 = np.mean(per_sample_f1)
        var_f1 = np.var(per_sample_f1)

        # Global macro metrics
        precision, recall, _, _ = precision_recall_fscore_support(
            filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
        )

        # Per-sample R-Precision
        r_precisions = []
        for true, pred in zip(true_labels, predicted_labels):
            if not true:
                continue
            k = len(true)
            top_k_pred = pred[:k]
            correct = len(set(top_k_pred) & set(true))
            r_precision = correct / k
            r_precisions.append(r_precision)

        mean_r_precision = np.mean(r_precisions) if r_precisions else 0.0
        var_r_precision = np.var(r_precisions) if r_precisions else 0.0

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": mean_f1,
            "F1 Variance": var_f1,
            "mRP": mean_r_precision,
            "mRP Variance": var_r_precision,
            "Length": len(true_labels)
        }

    def evaluate_results(self, results):
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            print(f"Precision: {metrics['Precision']:.4f}")
            print(f"Recall: {metrics['Recall']:.4f}")
            print(f"F1 Score: {metrics['F1 Score']:.4f} ± {metrics['F1 Variance']:.4f}")
            print(f"mRP: {metrics['mRP']:.4f} ± {metrics['mRP Variance']:.4f}")
            print(f"Length: {metrics['Length']}\n")


class OnlineTermsOfServiceDataset(Dataset):
    """
    Dataset for classifying fairness of online terms of service.
    It should be called as a generative task even though it is classification.
    """

    LABELS = ["clearly_fair", "potentially_unfair", "clearly_unfair"]
    LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
    INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

    def __init__(self):
        self.label_options = self.LABELS
        self.prompt = (
            "<|endoftext|>\n\n\nYou are a legal document fairness classifier. Above is a clause from an online Terms of Service document.\n"
            "Your task is to classify the fairness of the clause based on its content.\n"
            "- Only select one of the following labels:\n"
            "0: clearly fair\n"
            "1: potentially unfair\n"
            "2: clearly unfair\n"
            "Return **only the label number**.\n"
            "- Do not explain your answer or include any other text.\n"
        )

    def get_data(self, language, dataset_name, points_per_language):
        dataset = load_dataset("joelniklaus/online_terms_of_service", split='train')

        # Filter for language and non-empty fairness
        dataset = dataset.filter(lambda x: x.get("language") == language and x.get("unfairness_level") in self.LABELS)

        self.label_options = [0, 1, 2]
        # Build data list
        data = [{
            "text": item["sentence"],
            "labels": [self.LABEL_TO_INDEX[item["unfairness_level"]]]
        } for item in dataset]

        return data[:points_per_language], [], self.prompt

    def get_true(self, data):
        return [entry["labels"] for entry in data]

    def extract_labels_from_generated_text(self, generated_texts):
        all_labels = []
        for text in generated_texts:
            if not isinstance(text, str):
                all_labels.append([])
                continue
            matches = re.findall(r"\b\d+\b", text)
            if matches:
                all_labels.append([int(matches[0])])
            else:
                all_labels.append([])
        return all_labels

    def extract_label_indices(self, response: str):
        if not response or not isinstance(response, str):
            return None
        numbers = re.findall(r"\d+", response)
        return [int(numbers[0])] if numbers else None

    def evaluate(self, true_labels, predicted_labels):
        y_true = [lbl[0] if lbl else None for lbl in true]
        y_pred = [lbl[0] if lbl else None for lbl in predicted_labels]

        confusion_counter = Counter()
        penalties = []
        per_sample_f1 = []

        for true, pred in zip(y_true, y_pred):
            confusion_counter[(self.INDEX_TO_LABEL[true], self.INDEX_TO_LABEL[pred])] += 1

            # Penalty logic
            if true == pred:
                penalty = 0.0
            elif abs(true - pred) == 1:
                penalty = 0.5
            else:
                penalty = 1.0
            penalties.append(penalty)

            # Per-sample F1 logic
            true_vec = np.zeros(len(self.label_options))
            pred_vec = np.zeros(len(self.label_options))
            true_vec[true] = 1
            pred_vec[pred] = 1
            tp = np.sum(true_vec * pred_vec)
            precision_i = tp / np.sum(pred_vec) if np.sum(pred_vec) > 0 else 0.0
            recall_i = tp / np.sum(true_vec) if np.sum(true_vec) > 0 else 0.0
            f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0.0
            per_sample_f1.append(f1_i)

        mean_penalty = np.mean(penalties)
        var_penalty = np.var(penalties)
        mean_f1 = np.mean(per_sample_f1)
        var_f1 = np.var(per_sample_f1)

        # Global metrics
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1 Score": macro_f1,
            "F1 Variance": var_f1,
            "Penalty Mean": mean_penalty,
            "Penalty Variance": var_penalty,
            "Length": len(y_true),
            "Confusion Pairs": dict(confusion_counter)
        }

    def evaluate_results(self, results):
        output_path = "output/terms_of_service/results.txt"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for lang, metrics in results.items():
                f.write(f"Results for {lang}:\n")
                f.write(f"Precision: {metrics['Precision']:.4f}\n")
                f.write(f"Recall: {metrics['Recall']:.4f}\n")
                f.write(f"F1 Score: {metrics['F1 Score']:.4f} ± {metrics['F1 Variance']:.4f}\n")
                f.write(f"Penalty: {metrics['Penalty Mean']:.4f} ± {metrics['Penalty Variance']:.4f}\n")
                f.write(f"Length: {metrics['Length']}\n")
                f.write("Confusion Pairs:\n")
                for (true_label, pred_label), count in metrics["Confusion Pairs"].items():
                    f.write(f"  True: {true_label} → Pred: {pred_label}: {count}\n")
                f.write("\n")

        print(f"Evaluation results saved to {output_path}")


"""
Non Multilingual Dataset
"""
class CaseHOLD(Dataset):
    """
    Child class of Dataset representing the CaseHOLD dataset.
    """

    def __init__(self):
        self.label_options = ["A", "B", "C", "D", "E"]
        self.prompt = (
            "<|endoftext|> Question: Based on the case description, select the most appropriate legal answer by only "
            "stating the appropriate character:\n"
        )
        self.languages = ['en']

    def get_data(self, language=None):
        """
        Loads the CaseHOLD dataset.
        :return: the data and label options
        """
        dataset = load_dataset('lex_glue', 'case_hold', split='test')
        return self.extract_text(dataset)

    def extract_text(self, dataset):
        """
        Extracts and formats the data from the CaseHOLD dataset.
        :param dataset: the dataset containing the text data
        :return: a list of text data and labels
        """
        data = []
        count = 0
        print("Length of the dataset: ", len(dataset))
        for item in dataset:
            if count == 200:
                break
            count += 1

            # Create choices formatted with corresponding letters
            choices = "\n".join([f"{letter}) {ending}" for letter, ending in zip(self.label_options, item['endings'])])
            # Combine context and choices into the text
            text_with_choices = f"{item['context']}\n\n{choices}"

            data.append({
                "text": text_with_choices,  # Choices are now included in the text
                "label": item['label']  # Keep the label for evaluation
            })
        return data

    def get_true_labels(self, data):
        """
        :param data: list of data entries
        :return: list of true labels for the dataset
        """
        true_labels = [entry['label'] for entry in data]
        return true_labels

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using precision, recall, F1 score, and accuracy.
        :param true_labels: list of true labels
        :param predicted_labels: list of predicted labels
        """
        flat_predicted_labels = [item for sublist in predicted_labels for item in sublist]
        accuracy = accuracy_score(true_labels, flat_predicted_labels)

        print(f"Accuracy: {accuracy}")

    def extract_labels_from_generated_text(self, generated_text, label_options):
        """
        Extracts the first predicted label from the model's response.
        :param response: The model's output as a string
        :return: The first valid label (A, B, C, D, E) found in the response, or None if not found
        """
        # Find the first capital letter in the response within the range A-E
        print("Reached extract_labels in CaseHOLD class")
        match = re.search(r'\b([A-E])\b', generated_text)
        if match:
            print("Mathced response: ")
            print(match)
            return match.group(1)  # Return the first matched capital letter
        return ["F"]


    ###########################################################################################
    ################################## Non-Legal Datasets #####################################
    ###########################################################################################

class XQuAD(Dataset):
    """
    Child class of Dataset representing the XQuAD dataset.
    """

    def __init__(self, llm_judge):
        self.prompt = "<|endoftext|>\nTask: Given the question and the passage, extract the most relevant answer from the passage."
        self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.llm_judge = llm_judge


    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the XQuAD dataset from Hugging Face.

        :param language: The language for which data should be retrieved (ar, de, el, en, es, hi, ro, ru, th, tr, vi, zh)
        :param dataset_name: The dataset name ('google/xquad')
        :param points_per_language: Number of samples to return
        :return: Processed dataset and prompt
        """
        dataset = load_dataset("google/xquad", f"xquad.{language}", split="validation", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        inst = translate(language, self.prompt)
        self.lang = language
        return data, inst

    def extract_text(self, dataset, points_per_language):
        """
        Extracts passages, questions, and answers.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with context, question, and answers
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Passage: {item['context']}\nQuestion: {item['question']}",
                "answers": item["answers"]["text"],  # List of possible correct answers
                "question": item['question']
            })
        return data

    def get_true(self, data):
        """
        :return: A list of true answers for the dataset
        """
        return [entry["answers"] for entry in data]


    def evaluate(self, true_answers, predicted_answers, questions):
        """
        Evaluates the model's extracted answers using BLEU, METEOR, and Cosine Similarity,
        or with LLM-based scoring if self.llm_judge is enabled.

        :param true_answers: List of correct answers (each is a **list with one token**)
        :param predicted_answers: List of extracted answers (each is a full string)
        :param questions: List of questions (each is a **list with one token**)
        :return: Dictionary with BLEU, METEOR, and Cosine Similarity scores.
        """
        if len(true_answers) != len(predicted_answers):
            raise ValueError("true_answers and predicted_answers must have the same length")

        if self.llm_judge:
            prompts = []
            for true_list, pred_answer, question in zip(true_answers, predicted_answers, questions):
                true_answer = true_list[0] if true_list else ""
                pred_answer = pred_answer or ""
                question = question or ""

                prompt = (
                    "You are evaluating how well a generated answer responds to a given question. "
                    f"All content is in {get_language_from_code(self.lang)}. Use the real (true) answer as a reference to determine what a correct answer should look like. "
                    "Your task is to rate how well the generated answer answers the question, based on meaning and correctness, using the following scale:\n\n"
                    "5 - Fully answers the question with the same meaning as the real answer.\n"
                    "4 - Mostly answers the question with only minor differences from the real answer.\n"
                    "3 - Answers the question partially or includes significant inaccuracies.\n"
                    "2 - Barely answers the question or includes significant inaccuracies.\n"
                    "1 - Does not answer the question or is entirely incorrect.\n\n"
                    "Return only the number.\n\n"
                    f"Question: {question.strip()}\n"
                    f"Real answer (reference): {true_answer.strip()}\n"
                    f"Generated answer: {pred_answer.strip()}\n"
                    "Score:"
                )
                prompts.append(prompt)

            scores = self.llm_judge.judge(prompts)

            # Convert string scores to floats
            numeric_scores = []

            for raw_score in scores:
                if isinstance(raw_score, (int, float)):
                    numeric_scores.append(float(raw_score))
                    continue

                if not isinstance(raw_score, str):
                    numeric_scores.append(0.0)
                    continue

                # Search for the first valid number between 1 and 5
                match = re.search(r"\b([1-5](?:\.0)?)\b", raw_score)
                if match:
                    try:
                        numeric_scores.append(float(match.group(1)))
                    except ValueError:
                        numeric_scores.append(0.0)
                else:
                    numeric_scores.append(0.0)

            store_judge(scores, numeric_scores, self.lang)

            return {
                "LLM Similarity": np.mean(numeric_scores) if numeric_scores else 0.0
            }
        else:
            smoothing_function = SmoothingFunction().method1
            bleu_scores = []
            meteor_scores = []
            cosine_similarities = []

            for true_list, pred_answer in zip(true_answers, predicted_answers):
                # Extract true answer as a single string (it's wrapped in a list)
                true_answer = true_list[0] if true_list else ""

                # Handle empty predictions
                if not pred_answer or pred_answer.strip() == "":
                    bleu_scores.append(0.0)
                    meteor_scores.append(0.0)
                    cosine_similarities.append(0.0)
                    continue  # Skip further processing

                # Normalize: Remove newlines, trim spaces, lowercase
                true_answer = true_answer.strip().lower()
                pred_answer = pred_answer.strip().lower()

                try:
                    # Tokenize for BLEU and METEOR
                    tokenized_true = word_tokenize(true_answer)  # Single-token reference
                    tokenized_pred = word_tokenize(pred_answer)

                    # BLEU Score
                    bleu = sentence_bleu([tokenized_true], tokenized_pred, smoothing_function=smoothing_function)
                    bleu_scores.append(bleu)

                    # METEOR Score (NLTK expects a list of one reference, so we wrap it)
                    meteor = meteor_score([tokenized_true], tokenized_pred)
                    meteor_scores.append(meteor)

                except Exception as e:
                    print(f"Tokenization Error: {e}")
                    bleu_scores.append(0.0)
                    meteor_scores.append(0.0)

                # Cosine Similarity (BERT embeddings)
                try:
                    true_embedding = self.embedding_model.encode([true_answer])[0].reshape(1, -1)  # Ensure correct shape
                    pred_embedding = self.embedding_model.encode([pred_answer])[0].reshape(1, -1)

                    # Compute cosine similarity
                    cosine_sim = cosine_similarity(pred_embedding, true_embedding)[0][0]
                    cosine_similarities.append(cosine_sim)

                except Exception as e:
                    print(f"Embedding Error: {e}")
                    cosine_similarities.append(0.0)

            return {
                "BLEU Score": np.mean(bleu_scores) if bleu_scores else 0.0,
                "METEOR Score": np.mean(meteor_scores) if meteor_scores else 0.0,
                "Cosine Similarity": np.mean(cosine_similarities) if cosine_similarities else 0.0
            }



    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts the generated answers as predictions.

        :param generated_texts: List of generated answers
        :return: List of predicted answers
        """
        return generated_texts  # Directly return the generated answers as predictions

    def evaluate_results(self, results):
        """
        Prints BLEU, METEOR and Cosine Similarity scores for the dataset.
        """
        for lang, metrics in results.items():
            print(f"Results for {lang}:")
            if self.llm_judge:
                print(f"LLM Judge Similarity: {metrics['LLM Similarity']}")
            else:
                print(f"BLEU Score: {metrics['BLEU Score']}")
                print(f"METEOR Score: {metrics['METEOR Score']}")
                print(f"Cosine Similarity: {metrics['Cosine Similarity']}")


class XNLI(Dataset):
    """
    Child class of Dataset representing the XNLI dataset.
    """
    def __init__(self):
        self.label_options = ["0", "1", "2"]
        self.languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]
        self.prompt = ("<|endoftext|>\nTask: Please identify whether the premise entails or contradicts "
                       "the hypothesis, or neither. The answer should be '0' for entailment, "
                       "'1' for neither, or '2' for contradiction. The answer should be exactly '0', '1', or '2'."
                       )

    def get_data(self, language, dataset_name, points):
        """
        Loads the XNLI dataset for the specified language.
        :param language: the language of the dataset
        :return: the data and label options
        """
        dataset = load_dataset('xnli', language, split='test', trust_remote_code=True)
        self.language = language
        if language == 'all_languages':
            data = self.extract_text_all_languages(dataset)
        else:
            data = self.extract_text(dataset, points)
        return data, [], translate(language, self.prompt)

    def extract_text_all_languages(self, dataset):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data from all languages
        """
        data = []
        count = 0
        for item in dataset:
            if count == 5:
                break
            documents = item['text']
            texts = documents.keys()
            data.append({"text:": text, "labels": item['labels']} for text in texts)
            count += 1

    def extract_text(self, dataset, points):
        """
        :param dataset: the dataset containing the text data
        :return: a list of text data in the specified language
        """
        data = []
        count = 0
        for item in dataset:
            if count == points:
                break
            translator = GoogleTranslator(source="en", target=self.language)
            if self.language == "ar":
                text = item["hypothesis"] + translator.translate("Hypothesis: ") + item["premise"] + translator.translate("Premise: ")
            else:
                text = translator.translate("Premise: ") + item["premise"] + translator.translate(" Hypothesis: ") + item["hypothesis"]
            data.append({"text": text, "label": item['label']})
            count += 1
        return data

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using precision, recall, F1 score, and accuracy.
        :param true_labels: list of true labels
        :param predicted_labels: list of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        file_path = "XNLI_evaluation.csv"
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Language", "Accuracy"])
            writer.writerow([self.language, accuracy])

        print(f"Accuracy {self.language}: {accuracy}")

    import re

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts the first predicted label (0, 1, or 2) from the model's response.
        :param generated_texts: List of generated model outputs.
        :return: List of extracted labels (0, 1, 2), or None if no valid label is found.
        """
        word_to_digit = {"zero": 0, "one": 1, "two": 2}  # Handle word numbers
        all_labels = []

        print(generated_texts)

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                # Remove punctuation for easier matching
                text_lower = re.sub(r"[^\w\s]", "", text_lower)

                # Try to find exact numbers first
                match = re.findall(r"\b(0|1|2)\b", text_lower)

                if match:
                    all_labels.append(int(match[0]))  # Extract first match
                    continue  # Skip to next iteration

                # Try matching word numbers ("zero", "one", "two")
                for word, digit in word_to_digit.items():
                    if re.search(rf"\b{word}\b", text_lower):
                        all_labels.append(digit)
                        break  # Stop after first valid match
                else:
                    all_labels.append(None)  # No valid label found
            else:
                all_labels.append(None)

        return all_labels



    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]


"""
Non Multilingual Dataset
"""
class Go_Emotions(Dataset):
    """
    Child class of Dataset representing the GoEmotions dataset.
    """

    def __init__(self):
        self.label_options = [
            "admiration", "amusement", "anger", "annoyance", "approval",
            "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear",
            "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise"
        ]
        self.prompt = "<|endoftext|>" + (
                "Question: Which of the following emotions apply to this text? (You can select more than one): "
                + ', '.join(self.label_options) + " "
                                                  "Answer:"
        )

    def get_data(self, language=None):
        """
        Loads the GoEmotions dataset.
        :return: the data and label options
        """
        dataset = load_dataset('go_emotions', split='test')
        return self.extract_text(dataset)

    def extract_text(self, dataset):
        """
        Extracts and formats the data from the GoEmotions dataset.
        :param dataset: the dataset containing the text data
        :return: a list of text data and labels
        """
        data = []
        count = 0
        for item in dataset:
            if count == 50:
                break
            count += 1
            data.append({"text": item['text'], "labels": item['labels']})
        return data

    def get_true_labels(self, data):
        """
        :param data: list of data entries
        :return: list of true labels for the dataset
        """
        true_labels = [entry['labels'] for entry in data]
        return true_labels

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using precision, recall, and F1 score.
        :param true_labels: list of true labels
        :param predicted_labels: list of predicted labels
        """
        mlb = MultiLabelBinarizer(classes=list(range(len(self.label_options))))

        binary_true = mlb.fit_transform(true_labels)
        binary_pred = mlb.transform(predicted_labels)

        relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]
        filtered_binary_true = binary_true[:, relevant_labels]
        filtered_binary_pred = binary_pred[:, relevant_labels]

        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
        )

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")


"""
Datasets (non-multilingual) from decoding trust paper: https://arxiv.org/pdf/2306.11698
"""
class SST2(Dataset):
    """
    SST-2 dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1]
        self.prompt = "<|endoftext|>\nTask: Label the sentiment of the text as either negative or positive. The answer should be exact 'positive' or 'negative'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the SST-2 dataset from Hugging Face.

        :param language: Not needed for SST-2 (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "sst2", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts text and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with text and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({"text": item["sentence"], "label": item["label"]})
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}


    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts sentiment labels (positive or negative) from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for negative, 1 for positive)
        """
        all_labels = []

        for text in generated_texts:
            if text != None:
                text_lower = text.lower()

                if re.search(r"\bpositive\b", text_lower):
                    all_labels.append(1)
                elif re.search(r"\bnegative\b", text_lower):
                    all_labels.append(0)
                else:
                    all_labels.append(None)
            else:
                all_labels.append(None)
        return all_labels

    def get_true(self, data):
        """
        :return: a list of true labels for the dataset
        """
        print(data)
        true_labels = [entry['label'] for entry in data]
        return true_labels

    def evaluate_results(self, results):
        # Print out the results for each language
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "negative"
            if entry["label"] == 1:
                entry["label"] = "positive"
        return new_data

class QQP(Dataset):
    """
    QQP dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1]  # 0: Not duplicate, 1: Duplicate
        self.prompt = "<|endoftext|>\nTask: Please identify whether Question 1 has the same meaning as Question 2. The answer should be exact 'yes' or 'no'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the QQP dataset from Hugging Face.

        :param language: Not needed for QQP (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "qqp", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts question pairs and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with question pairs and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Question 1: {item['question1']}, Question 2: {item['question2']}",
                "label": item["label"]
            })
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts duplicate/not duplicate labels from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for not duplicate, 1 for duplicate)
        """
        all_labels = []

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                if re.search(r"\byes\b", text_lower):
                    all_labels.append(1)
                elif re.search(r"\bno\b", text_lower):
                    all_labels.append(0)
                else:
                    all_labels.append(None)
            else:
                all_labels.append(None)
        return all_labels

    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

    def evaluate_results(self, results):
        """
        Prints accuracy and results for each language (even though QQP is monolingual).
        """
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "no"
            if entry["label"] == 1:
                entry["label"] = "yes"
        return new_data

class MNLI(Dataset):
    """
    MNLI dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1, 2]  # 0: Contradiction, 1: Neutral, 2: Entailment
        self.prompt = "<|endoftext|>\nTask: Please identify whether the premise entails or contradicts the hypothesis, or neither. The answer should be exactly 'entailment', 'neutral', or 'contradiction'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the MNLI dataset from Hugging Face.

        :param language: Not needed for MNLI (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "mnli", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts premise-hypothesis pairs and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with premise, hypothesis, and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Premise: {item['premise']}, Hypothesis: {item['hypothesis']}",
                "label": item["label"]
            })
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts contradiction/neutral/entailment labels from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for contradiction, 1 for neutral, 2 for entailment)
        """
        all_labels = []

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                if re.search(r"\bentailment\b", text_lower):
                    all_labels.append(0)
                elif re.search(r"\bneutral\b", text_lower):
                    all_labels.append(1)
                elif re.search(r"\bcontradiction\b", text_lower):
                    all_labels.append(2)
                else:
                    all_labels.append(None)
            else:
                all_labels.append(None)
        return all_labels

    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

    def evaluate_results(self, results):
        """
        Prints accuracy and results for each language (even though MNLI is monolingual).
        """
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "entailment"
            if entry["label"] == 1:
                entry["label"] = "neutral"
            if entry["label"] == 2:
                entry["label"] = "contradiction"
        return new_data

class QNLI(Dataset):
    """
    QNLI dataset from the GLUE benchmark.
    """

    def __init__(self):
        self.label_options = [0, 1]  # 0: Entailment, 1: Not Entailment
        self.prompt = "<|endoftext|>\nTask: Determine whether the sentence answers the question. The answer should be exactly 'yes' or 'no'."

    def get_data(self, language, dataset_name, points_per_language):
        """
        Loads the QNLI dataset from Hugging Face.

        :param language: Not needed for QNLI (single language)
        :param dataset_name: The dataset name (GLUE benchmark)
        :param points_per_language: Number of samples to return
        :return: Processed dataset, label options, and prompt
        """
        dataset = load_dataset("glue", "qnli", split="train", trust_remote_code=True)
        data = self.extract_text(dataset, points_per_language)
        return data, self.label_options, self.prompt

    def extract_text(self, dataset, points_per_language):
        """
        Extracts question-passage pairs and labels.

        :param dataset: The dataset containing text data
        :return: List of dictionaries with question, passage, and labels
        """
        data = []
        for i, item in enumerate(dataset):
            if i >= points_per_language:
                break
            data.append({
                "text": f"Question: {item['question']}, Sentence: {item['sentence']}",
                "label": item["label"]
            })
        return data

    def get_true_labels(self, data):
        """
        :return: List of true labels
        """
        return [entry["label"] for entry in data]

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model using accuracy.

        :param true_labels: List of true labels
        :param predicted_labels: List of predicted labels
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        return {"Accuracy": accuracy}

    def extract_labels_from_generated_text(self, generated_texts):
        """
        Extracts entailment/not entailment labels from generated text.

        :param generated_texts: List of generated text responses
        :return: List of extracted labels (0 for entailment, 1 for not entailment)
        """
        all_labels = []

        for text in generated_texts:
            if text is not None:
                text_lower = text.lower()

                if re.search(r"\byes\b", text_lower):
                    all_labels.append(0)
                elif re.search(r"\bno\b", text_lower):
                    all_labels.append(1)
                else:
                    all_labels.append(None)  # Can't determine
            else:
                all_labels.append(None)

        return all_labels

    def get_true(self, data):
        """
        :return: A list of true labels for the dataset
        """
        return [entry['label'] for entry in data]

    def evaluate_results(self, results):
        """
        Prints accuracy and results for each language (even though QNLI is monolingual).
        """
        for lang, metric in results.items():
            print(f"Results for {lang}:")
            print(f"Accuracy: {metric['Accuracy']}")

    def get_mapped_data(self, data):
        new_data = copy.deepcopy(data)
        for entry in new_data:
            if entry["label"] == 0:
                entry["label"] = "yes"
            if entry["label"] == 1:
                entry["label"] = "no"
        return new_data



