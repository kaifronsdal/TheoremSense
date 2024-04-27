import pickle
import re


def save_arguments(args, path):
    """Saves argparse arguments to a pickle file.
  
    Args:
      args: The argparse namespace containing the arguments to save.
      path: The path to the file to save the arguments to.
    """
    with open(path, 'wb') as f:
        pickle.dump(args, f)


def load_arguments(path):
    """Loads argparse arguments from a pickle file.

    Args:
      path: The path to the file containing the saved arguments.

    Returns:
      The loaded argparse namespace, or None if the file does not exist.
    """
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def matching_brackets(string, idx, open_char='{', close_char='}'):
    assert string[idx] == open_char
    if idx < len(string):
        opening = [i for i, c in enumerate(string[idx + 1:]) if c == open_char]
        closing = [i for i, c in enumerate(string[idx + 1:]) if c == close_char]
        for i, j in enumerate(closing):
            if i >= len(opening) or j < opening[i]:
                return j + idx + 1
    return -1


def get_boxed_answer(answer: str):
    match = re.search(r'\\boxed{', answer)
    boxed = ''
    if match is not None:
        boxed_start = match.end() - 1
        boxed_end = matching_brackets(answer, boxed_start)
        boxed = answer[boxed_start + 1:boxed_end]
    else:
        match = re.search(r'\\boxed ', answer)
        if match is not None:
            boxed = answer[match.end()]
    return boxed
