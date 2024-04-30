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

import random

def generate_boolean_expression(num_terms=3):
    operators = ['and', 'or']
    values = ['True', 'False']
    expression = []

    # Start with a random boolean value
    expression.append(random.choice(values))

    # Add operators and boolean values
    for _ in range(num_terms - 1):
        operator = random.choice(operators)
        value = random.choice(values)
        expression.append(operator)
        expression.append(value)

    # Join all parts to form the final expression
    expression_str = ' '.join(expression)
    return expression_str, eval(expression_str)

def generate_bool_expression(
    num_groups: int = 3,
    num_terms: int = 4,
    and_false: bool = False,
    or_true: bool = False,
    randoms: bool = False,
):
    if and_false == False and or_true == False and randoms == False:
        choice = random.choice(['False', 'True'])
        if choice == "False":
            and_false = True
        else:
            or_true = True
    
    expression = []

    for _ in range(num_groups):
        # Determine the number of terms in this group
        num_terms = random.randint(2, num_terms)
        sub_expr, _ = generate_boolean_expression(num_terms)

        # Add parentheses around the sub-expression
        if len(expression) > 0:
            operator = random.choice(['and', 'or'])
            expression.append(operator)
        expression.append(f"({sub_expr})")

    # Join all parts to form the final expression
    expression_str = ' '.join(expression)

    if and_false:
        expression_str = "(" + expression_str + ")" + " and False"
    elif or_true:
        expression_str = expression_str + ' or True'

    return expression_str, eval(expression_str)

def generate_relation_problem(
    n: int = 10,
):
    # Generate random cities excluding A and Z
    cities = [chr(i) for i in range(65, 65 + n) if chr(i) not in ['A', 'Z']]
    random.shuffle(cities)
    
    # Initialize connections and make sure A and Z are included
    connections = []
    a_connection = random.choice(cities)
    connections.append(('A', a_connection))
    
    # Randomly connect other cities
    for i in range(len(cities) - 1):
        connections.append((cities[i], cities[i + 1]))
    
    # Randomly decide if A should be connected to Z through the point A is connected to
    if random.choice([True, False]):
        connections.append((a_connection, 'Z'))
        answer = True
    else:
        answer = False

    text = ""
    
    for each in connections:
        text += each[0] + " is connected with " + each[1] + "\n"
    
    return text, answer

import numpy as np

def generate_linear_equations(
    num_vars: int = 2, 
    num_equations: int = 2, 
    num_related: int = 1,
):
    """
    生成一个线性方程组以及线性相关的方程。

    参数:
    - num_vars: 变量的数量
    - num_equations: 初始生成的独立方程的数量
    - num_related: 要添加的与现有方程线性相关的方程数量

    返回:
    - equations: 生成的所有方程的字符串列表
    - solutions: 对应变量的解字典
    """
    np.random.seed(0)  # 设置随机种子

    # 生成随机整数系数矩阵
    coeff_matrix = np.random.randint(-10, 10, (num_equations, num_vars))

    # 变量名列表，如x1, x2, x3, ...
    variables = [f"x{i + 1}" for i in range(num_vars)]

    # 创建原始方程
    equations = []
    for row in coeff_matrix:
        equation = " + ".join(f"{coef}*{var}" if coef != 0 else "" for coef, var in zip(row, variables)).strip("+ ").replace("+ -", "- ")
        equation += " = 0"
        equations.append(equation)

    # 添加线性相关的方程
    for _ in range(num_related):
        # 随机选择一个已有方程并乘以一个随机系数生成新的方程
        index = np.random.randint(0, num_equations)
        multiplier = np.random.randint(1, 5)
        new_equation = " + ".join(f"{multiplier * coef}*{var}" if multiplier * coef != 0 else "" for coef, var in zip(coeff_matrix[index], variables)).strip("+ ").replace("+ -", "- ")
        new_equation += " = 0"
        equations.append(new_equation)

    # 所有变量假设解为0
    solutions = {var: 0 for var in variables}

    return equations, solutions
