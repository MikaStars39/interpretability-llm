import random

def generate_sum_expressions(
    min_terms: int = 1,
    max_terms: int = 7,
    num_range: int = 10,
    need_total_sum: int = 0,
    minus: bool = True,
    ):
    terms = []
    sum_so_far = 0
    num_terms = random.randint(min_terms, max_terms)
    if need_total_sum is None:
        need_total_sum=random.randint(-num_range, num_range) if minus else random.randint(0, num_range)
    if num_terms == 1:
        expression = str(need_total_sum)
        sum_so_far = need_total_sum
    else:
        for i in range(num_terms - 1):
            number = random.randint(-num_range, num_range) if minus else random.randint(0, num_range)
            op = '+' if number >= 0 else "-"
            sum_so_far += number
            if i == 0 and op == '+':
                op = ''
            terms.append(f'{op}{abs(number)}')
        final_number = need_total_sum-sum_so_far
        final_op = '+' if final_number >= 0 else '-'
        terms.append(f'{final_op}{abs(final_number)}')
        expression = ''.join(terms)
    return expression, need_total_sum

# print(generate_sum_expressions(
#         min_terms=1,
#         max_terms=3,
#         num_range=10,
#         need_total_sum=0,
#     ))
