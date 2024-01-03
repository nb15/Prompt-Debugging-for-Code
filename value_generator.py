import random
import string

def regenerate_values_dynamically(nested_list):
    # Helper function to flatten a nested list
    def flatten_list(nlist):
        for i in nlist:
            if isinstance(i, list):
                yield from flatten_list(i)
            elif isinstance(i, tuple):
                yield from flatten_list(list(i))
            elif isinstance(i, dict):
                for value in i.values():
                    if isinstance(value, (int, float, list, tuple, dict)):
                        yield from flatten_list([value])
                    else:
                        yield value
            else:
                yield i

    # Function to generate random strings
    def random_string(length, char_pool):
        return ''.join(random.choice(char_pool) for _ in range(length))

    # Function to determine the type of string to regenerate
    def regenerate_string(s, min_len, max_len):
        length = random.randint(min_len, max_len)
        if ' ' in s:
            words = s.split()
            regenerated_words = [random_string(length, string.ascii_letters) for word in words]
            return ' '.join(regenerated_words)
        elif s.islower():
            return random_string(length, string.ascii_lowercase)
        elif s.isupper():
            return random_string(length, string.ascii_uppercase)
        elif s.istitle():
            return random_string(length, string.ascii_letters).title()
        else:
            return random_string(length, string.ascii_letters)

    # Flattening the list to find the min and max values and string lengths
    flat_list = list(flatten_list(nested_list))
    strings = [s for s in flat_list if isinstance(s, str)]

    # Check if strings list is not empty
    if strings:
        min_len, max_len = min(map(len, strings)), max(map(len, strings))
    else:
        min_len, max_len = 1, 10  # Default values if no strings are found

    min_value, max_value = min((x for x in flat_list if isinstance(x, (int, float))), default=0), max((x for x in flat_list if isinstance(x, (int, float))), default=0)

    # Function to replace values in the nested list
    def replace_values(sublist):
        if isinstance(sublist, list):
            return [replace_values(item) for item in sublist]
        elif isinstance(sublist, tuple):
            return tuple(replace_values(list(sublist)))
        elif isinstance(sublist, dict):
            return {key: replace_values(value) for key, value in sublist.items()}
        else:
            if isinstance(sublist, int):
                return random.randint(int(min_value), int(max_value))
            elif isinstance(sublist, float):
                return random.uniform(min_value, max_value)
            elif isinstance(sublist, str):
                return regenerate_string(sublist, min_len, max_len)

    return replace_values(nested_list)

# Example usage
# example_list_1 = [[[[1, 2], [2, 3]], [[3, 4], [5, 7]]], [[[3, 4], [5, 6]]]]
# example_list_2 = [2, 4, 0, 7, 8, 7]
# example_list_3 = [[1, 2, -3], [2, -4, 5], [1, 1, 1], 8, 9]
# example_list_4 = [[1, 2.5, -3], [2, -4, 5.1], [1, 1.4, 1]]
# example_list_5 = [(8, 2, 3, -1, 7)]
# example_list_6 = [(3, 5), (1, 7), (10, 3), (1, 2, [2, 4, 5]), [1, 2, 3, 4, 5], 4, 5.5]
# example_list_7 = [('Hi', [2, 3.3]),('dello', [4, 9.43]), ('WORLD Dog', [1, 2.3, 4.5]), ('BIG', [1, 2, 3, 4, 5])]
# example_list_8 =  [{'a': 100, 'b': 200,'c':300},{'a': 300, 'b': 200, 'd':400}]
# example_list_9 = [{1:'python',2:'java'}]

# regenerated_list_1 = regenerate_values_dynamically(example_list_1)
# regenerated_list_2 = regenerate_values_dynamically(example_list_2)
# regenerated_list_3 = regenerate_values_dynamically(example_list_3)
# regenerated_list_4 = regenerate_values_dynamically(example_list_4)
# regenerated_list_5 = regenerate_values_dynamically(example_list_5)
# regenerated_list_6 = regenerate_values_dynamically(example_list_6)
# regenerated_list_7 = regenerate_values_dynamically(example_list_7)
# regenerated_list_8 = regenerate_values_dynamically(example_list_8)
# regenerated_list_9 = regenerate_values_dynamically(example_list_9)

# print('Original list 1:', example_list_1)
# print('Regenerated list 1:', regenerated_list_1, '\n')

# print('Original list 2:', example_list_2)
# print('Regenerated list 2:', regenerated_list_2, '\n')

# print('Original list 3:', example_list_3)
# print('Regenerated list 3:', regenerated_list_3, '\n')

# print('Original list 4:', example_list_4)
# print('Regenerated list 4:', regenerated_list_4, '\n')

# print('Original list 5:', example_list_5)
# print('Regenerated list 5:', regenerated_list_5, '\n')

# print('Original list 6:', example_list_6)
# print('Regenerated list 6:', regenerated_list_6, '\n')

# print('Original list 7:', example_list_7)
# print('Regenerated list 7:', regenerated_list_7, '\n')

# print('Original list 8:', example_list_8)
# print('Regenerated list 8:', regenerated_list_8, '\n')

# print('Original list 9:', example_list_9)
# print('Regenerated list 9:', regenerated_list_9, '\n')