import json
import random
import string

def generate_kv_pairs(
    length: int = 100,
    num_pairs_per_position: int = 1000,
    ):
    filename = f"kv_pairs_{num_pairs_per_position}_{length}.json"
    kv_pairs = {}
    for position in range(num_pairs_per_position):
        all_queries = []
        for _ in range(length):
            key = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            value = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
            query = "Key: " + key + ", Value: " + value 
            all_queries.append(query)
        kv_pairs[position] = all_queries
    print(f'Successfully saved {num_pairs_per_position} KV pairs to {filename}')
    save_to_json(kv_pairs, filename)

def save_to_json(kv_pairs, filename):
    with open(filename, 'w') as f:
        json.dump(kv_pairs, f)


generate_kv_pairs(
    length=100,
    num_pairs_per_position=100
)
    
    


