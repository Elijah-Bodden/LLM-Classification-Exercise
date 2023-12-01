# Generate examples for the various categorization tasks

import random
import pandas as pd

def get_n_from_file(filename, n, positive):
    # Positive used as a hacky way to change which part we sample from between the two sampling times to prevent repeat examples
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        return lines[:n] if positive else lines[-n:]

def util_rand_except(exception, start, end):
    num = random.randint(start, end)
    while num == exception:
        num = random.randint(start, end)
    return num

def rot_n(input, n):
    output = ""
    for c in input:
        if c.isalpha():
            if c.isupper():
                output += chr((ord(c) - 65 + n) % 26 + 65)
            else:
                output += chr((ord(c) - 97 + n) % 26 + 97)
        else:
            output += c
    return output

def batch_ordinary(batch_size, positive, func):
    return [func(positive) for _ in range(batch_size)]

# Ordinary (no external input) generators:

def long_short_random(positive):
    # True if the input is long (> 4 characters), false if it's short
    if positive:
        return ''.join([str(chr(random.randint(97, 122))) for _ in range(random.randint(10, 15))])
    else:
        return ''.join([str(chr(random.randint(97, 122))) for _ in range(random.randint(1, 4))])

def tiled_letter(positive):
    # True iff the letter is repeated the same number of times as its position in the alphabet
    letter_idx = random.randint(97, 122)
    letter = str(chr(letter_idx))
    if positive:
        return letter * (letter_idx - 96)
    else:
        return letter * util_rand_except(letter_idx - 96, 1, 26)

def palindrome(positive):
    # True iff the word is a palindrome
    word = ''.join([str(chr(random.randint(97, 122))) for _ in range(5)])
    if positive:
        return word + word[::-1]
    else:
        return word + ''.join([str(chr(random.randint(97, 122))) for _ in range(5)])
    
def eq_3_mod_8(positive):
    # True iff the number is equal to 3 mod 8
    num = random.randint(0, 100)
    if positive:
        return num * 8 + 3
    else:
        return num * 8 + util_rand_except(3, 0, 7)

def emoji_strictly_increasing(positive):
    # True iff the emoji codepoints are strictly increasing
    emoji = [str(chr(random.randint(0x1F600, 0x1F64F))) for _ in range(10)]
    if positive:
        return "".join(sorted(emoji))
    else:
        return "".join(emoji)

# Alternative if emoji is too hard
def int_strictly_increasing(positive):
    # True iff the integers are strictly increasing
    ints = [str(random.randint(0, 10)) for _ in range(4)]
    if positive:
        return "".join(sorted(ints))
    else:
        return "".join([str(i) for i in ints])

def random_string_len_is_odd(positive):
    # True iff the length of a random string is odd
    string = ''.join([str(chr(random.randint(97, 122))) for _ in range(random.randint(0, 10))])
    if positive and len(string) % 2 == 0:
        return string + str(chr(random.randint(97, 122)))
    if not positive and len(string) % 2 == 1:
        return string[:-1]
    return string
    
def all_left_glyphs(positive):
    # True iff the convex side of the characters in the string all face left.
    left_glyphs = ["<", "(", "[", "{"]
    right_glyphs = [">", ")", "]", "}"]
    sample_left = lambda: [left_glyphs[random.randint(0, 3)] for _ in range(4)]
    sample_right = lambda: [right_glyphs[random.randint(0, 3)] for _ in range(4)]
    if positive:
        x = sample_left() + sample_left()
        random.shuffle(x)
        return "".join(x)
    else:
        x = sample_left() + sample_right()
        random.shuffle(x)
        return "".join(x)

def batch_from_file_unpaired(batch_size, positive, func, filename):
    return [func(line, positive) for line in get_n_from_file(filename, batch_size, positive)]

# Generators based off of modifying a single kind of input:

def sentence_rot_13_or_1(sentence, positive):
    # True if the string is a rot13 sentence, false if it's rot1
    if positive:
        return rot_n(sentence, 13)
    else:
        return rot_n(sentence, 1)
            
def sentence_is_punctuated(sentence, positive):
    # True iff the sentence is punctuated
    punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    if positive:
        return sentence
    else:
        return "".join([c for c in sentence if c not in punctuation])

def sentence_random_chars_dropped(sentence, positive):
    # True iff the sentence has random characters dropped
    if positive:
        return sentence
    else:
        drop_count = random.randint(0, 4)
        drop_indices = random.sample(range(len(sentence)), drop_count)
        return "".join([c for i, c in enumerate(sentence) if i not in drop_indices])

def all_upper(sentence, positive):
    # True iff the sentence has all vowels capitalized
    if positive:
        return sentence.upper()
    else:
        return sentence

def brand_has_one_letter_different(brand, positive):
    # True iff the brand name has one letter different from the original
    if positive:
        brand = list(brand)
        idx = random.randint(0, len(brand) - 1)
        brand[idx] = str(chr(util_rand_except(ord(brand[idx]), 97, 122)))
        return "".join(brand)
    else:
        return brand
    
def batch_from_file_paired(batch_size, positive, filename):
    # Read from existing pairs in the target (csv) file
    names = pd.read_csv(filename, encoding='utf-8')
    if positive:
        return list(names['t'][:batch_size])
    else:
        return list(names['f'][:batch_size])