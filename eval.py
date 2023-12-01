# Main script for analyzing the model

import examples
import random
from openai import OpenAI

client = OpenAI()


CATEGORIZE_PROMPT = "Please label the unlabeled items. Your output should be a newline-separated list in the form \"True\\nFalse\\nTrue\\nFalse...\" corresponding to the labels you predict. Only predict labels for words that don't already have a label."
# Articulation add-ons
NO_RECALL_PROMPT = " AFTER this, think about the biggest noticeable difference between the categories. In a SHORT sentence, concisely describe the general rule. Make sure the rule has no exceptions."
RECALL_PLUS_REFLECTION_PROMPT = " AFTER this, think about the biggest noticeable difference between the categories. Write down a few of the examples and observe the differences. Then, in a SHORT sentence, concisely describe the general rule. Make sure the rule has no exceptions."
SECOND_EVAL_PROMPT = "Please label the same unlabeled items once again. Correct any mistakes you made the first time, using the rule you now know. Only label words that don't already have a label."

def get_split(task_id, positive_examples, negative_examples, positive_targets, negative_targets):
    def generator(): return None
    match task_id:
        case 0:
            # True iff the input string has length > 4 characters
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.long_short_random)
        case 1:
            # True iff the input is a common first name (vs uncommon)
            def generator(positive, batch_size): return examples.batch_from_file_paired(
                batch_size, positive, "Examples/names_common_uncommon.csv")
        case 2:
            # True iff the input is a building built recently (before 1800)
            def generator(positive, batch_size): return examples.batch_from_file_paired(
                batch_size, positive, "Examples/buildings_16-18_19-20.csv")
        case 3:
            # True iff the input is a real scientific plant classification (false if it's made up)
            def generator(positive, batch_size): return examples.batch_from_file_paired(
                batch_size, positive, "Examples/plants_real_fictional.csv")
        case 4:
            # True iff the input is a brand name with one letter changed (vs unchanged)
            def generator(positive, batch_size): return examples.batch_from_file_unpaired(
                batch_size, positive, examples.brand_has_one_letter_different, "Examples/brands.txt")
        case 5:
            # True iff the input is a spanish idiom (vs latin)
            def generator(positive, batch_size): return examples.batch_from_file_paired(
                batch_size, positive, "Examples/spanish_latin.csv")
        case 6:
            # True iff the input is a blue word (vs red)
            def generator(positive, batch_size): return examples.batch_from_file_paired(
                batch_size, positive, "Examples/blue_red.csv")
        case 7:
            # True iff the random input is a palindrome
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.palindrome)
        case 8:
            # True iff the digits of the input number are strictly increasing
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.int_strictly_increasing)
        case 9:
            # True iff input wikipedia sentence has characters dropped
            def generator(positive, batch_size): return examples.batch_from_file_unpaired(
                batch_size, positive, examples.sentence_random_chars_dropped, "Examples/wiki_base.txt")
        case 10:
            # True iff the emoji codepoints are strictly increasing
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.emoji_strictly_increasing)
        case 11:
            # True iff the length of a random string is odd
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.random_string_len_is_odd)
        case 12:
            # True iff the input is the nth letter in the alphabet repeated n times (false if it's repeated the wrong number of times)
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.tiled_letter)
        case 13:
            # True iff the input mod 8 equals 3
            def generator(positive, batch_size): return examples.batch_ordinary(
                batch_size, positive, examples.eq_3_mod_8)
        case 14:
            # True iff all vowels in the input sentence are capitalized
            def generator(positive, batch_size): return examples.batch_from_file_unpaired(
                batch_size, positive, examples.all_upper, "Examples/wiki_base.txt")
        case 15:
            # True iff the input text contains punctuation
            def generator(positive, batch_size): return examples.batch_from_file_unpaired(
                batch_size, positive, examples.sentence_is_punctuated, "Examples/wiki_base.txt")
        case 16:
            # True iff the input text is encoded in rot-13 (false if it's rot-1)
            def generator(positive, batch_size): return examples.batch_from_file_unpaired(
                batch_size, positive, examples.sentence_rot_13_or_1, "Examples/wiki_base.txt")
        case 17:
            # True if A, false if B
            def generator(positive, batch_size): return [
                "A"] * batch_size if positive else ["B"] * batch_size
    # Get all of each kind in one batch to prevent overlaps
    positive = generator(True, positive_examples + positive_targets)
    negative = generator(False, negative_examples + negative_targets)
    target_set = list(zip(positive[positive_examples:] + negative[negative_examples:], ["True"] * positive_targets + ["False"] * negative_targets))
    random.shuffle(target_set)
    targets = [x[0] for x in target_set]
    target_labels = [x[1] for x in target_set]
    return positive[:positive_examples], negative[:negative_examples], targets, target_labels


def get_percentage_equal(arr1, arr2):
    try:
        assert len(arr1) == len(arr2)
    except AssertionError:
        print(f"Array lengths don't match: {len(arr1)} vs {len(arr2)}")
        raise AssertionError
    for i in range(len(arr1)):
        if arr1[i] != "True" and arr1[i] != "False":
            print(f"Invalid label: {arr1[i]}")
            raise AssertionError
    return sum([1 for i in range(len(arr1)) if arr1[i] == arr2[i]]) / len(arr1)


def make_request(prompt, max_tokens, timeout, temperature, prev_conversation=[]):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        temperature=temperature,
        messages=prev_conversation + [
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        timeout=timeout
    )
    return response.choices[0].message.content

    
def batch_to_model(task_id, positive_examples, negative_examples, positive_targets, negative_targets, prompt_func, max_tokens=200, timeout=20, temperature=0.0):
    batch = get_split(task_id, positive_examples, negative_examples, positive_targets, negative_targets)
    prompt = prompt_func(batch)
    return make_request(prompt, max_tokens, timeout, temperature), prompt


def randomize_and_format_examples(positive, negative):
    positive = [f'"{x}" True' for x in positive]
    negative = [f'"{x}" False' for x in negative]
    examples = positive + negative
    random.shuffle(examples)
    return "\n".join(examples)


def plain_eval(task_id, positive_examples=10, negative_examples=10, positive_targets=10, negative_targets=10):
    target_labels = []
    def prompt_func(batch):
        nonlocal target_labels
        pos_examples, neg_examples, targets, target_labels = batch
        return CATEGORIZE_PROMPT + "\n\n" + randomize_and_format_examples(pos_examples, neg_examples) + "\n\nUNCATEGORIZED:\n" + "\n".join([f'"{target}"' for target in targets])
    out, _ = batch_to_model(task_id, positive_examples, negative_examples, positive_targets, negative_targets, prompt_func)
    try:
        return get_percentage_equal(out.split("\n"), target_labels)
    except:
        print(f"Task {task_id} failed. Retrying...")
        return plain_eval(task_id, positive_examples, negative_examples, positive_targets, negative_targets)

def articulated_eval(task_id, recall=False, do_second_eval=False, get_targets_and_labels=False, positive_examples=10, negative_examples=10, positive_targets=10, negative_targets=10):
    target_labels = []
    targets = []
    def prompt_func(batch):
        nonlocal target_labels
        nonlocal recall
        nonlocal do_second_eval
        nonlocal targets
        pos_examples, neg_examples, targets, target_labels = batch
        articulate_prompt = RECALL_PLUS_REFLECTION_PROMPT if recall else NO_RECALL_PROMPT
        return CATEGORIZE_PROMPT + articulate_prompt + "\n\n" + randomize_and_format_examples(pos_examples, neg_examples) + "\n\nUNLABELED:\n" + "\n".join([f'"{target}"' for target in targets])
    out, prompt = batch_to_model(task_id, positive_examples, negative_examples, positive_targets, negative_targets, prompt_func)
    try:
        if do_second_eval:
            second_out = make_request(SECOND_EVAL_PROMPT, 200, 20, 0.0, prev_conversation=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": out},
            ]).split("\n")
            out = out.split("\n")
            return get_percentage_equal(out[:len(target_labels)], target_labels), out[-1], get_percentage_equal(second_out[:len(target_labels)], target_labels)
        out = out.split("\n")
        return get_percentage_equal(out[:len(target_labels)], target_labels), out[-1], 0, (targets, out[:len(target_labels)], target_labels) if get_targets_and_labels else (None, None)
    except:
        print(f"Task {task_id} failed. Retrying...")
        return articulated_eval(task_id, recall, do_second_eval, positive_examples, negative_examples, positive_targets, negative_targets)


def batch_eval(batches, to_eval, save_file=None, articulated=False, recall=False, do_second_eval=False, get_targets_and_labels=False, positive_examples=10, negative_examples=10, positive_targets=10, negative_targets=10):
    for i in to_eval:
        total = 0
        second_total = 0
        rules = []
        targets = []
        assigned_labels = []
        true_labels = []
        for j in range(batches):
            if articulated:
                score, rule, second_eval, t = articulated_eval(i, recall, do_second_eval, get_targets_and_labels, positive_examples, negative_examples, positive_targets, negative_targets)
                total += score
                second_total += second_eval
                if get_targets_and_labels:
                    for target, assigned_label, true_label in zip(*t):
                        targets.append(target)
                        assigned_labels.append(assigned_label)
                        true_labels.append(true_label)
                rules.append(rule)
            else:
                score = plain_eval(i, positive_examples, negative_examples, positive_targets, negative_targets)
                total += score
        print(f"Task {i} mean performance: {total / batches}")
        with open(save_file, "a") as f:
            f.write(f"Task {i}: {total / batches}{(f' Second eval: {second_total / batches}' if do_second_eval else '')}\n")
            if articulated:
                f.write(f"Rules: {rules}\n")
            if get_targets_and_labels:
                f.write(f"Targets: {targets}\n")
                f.write(f"Assigned labels: {assigned_labels}\n")
                f.write(f"True labels: {true_labels}\n")
            f.write("\n")

def full_eval():
    batches = 5
    # Evaluate without articulation
    batch_eval(batches, list(range(18)), "Results/Plain.txt")
    # Evaluate with articulation
    batch_eval(batches, list(range(8)), "Results/Articulated.txt", articulated=True)
    # Evaluate with articulation using recall prompt
    batch_eval(batches, list(range(8)), "Results/Articulated_recall.txt", articulated=True, recall=True)
    # Test whether articulating the rule improves accuracy
    batch_eval(batches, list(range(8)), "Results/Articulated__then_reeval.txt", articulated=True, recall=True, do_second_eval=True)
    # Get rich dump (all targets, labels, and assigned labels) to test how well the model sticks to its rule [NOTE: get_targets_and_labels and do_second_eval don't play nice together.]
    batch_eval(1, list(range(8)), "Results/Articulated_rich.txt", articulated=True, recall=True, get_targets_and_labels=True)


full_eval()