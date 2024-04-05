import json
import random
import string

def generate_kv(
    data_size:int = 1,
    length:int = 49,
):
    kv_pairs = []
    for ids in range(length):
        all_queries = "I will give you some keys and values. Then I ask you a key, you respones me with its value."
        example = " Example: Key: 2h3d73fs, Value: d3d83fs8. The key is 2h3d73fs, so the value is d3d83fs8. "
        all_queries += example
        all_values = []
        for pos in range(length):
            key = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            value = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            if ids == pos:
                ans_key = key
                ans_value = value
            query = " Key: " + key + ", Value: " + value + ";"
            all_queries += query
        all_queries = all_queries + " Question: The key is " + ans_key + " , so the value is "
        all_values.append(ans_value)
        kv_pairs.append((ans_value, all_queries))
    return kv_pairs

    


