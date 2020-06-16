from random import randint

operators = "+-*/"


def complete_eq(formula: str) -> str:
    """
    evaluate the formula
    :param formula:
    :return:
    """
    result = eval(formula)
    return "".join([formula, "=", str(result), "#"])


def simple_eq(max_range: int = 100) -> str:
    """
    generate simple equation with number in the
    :param max_range: max range of the number in the formula (not the result)
    :return:
    """
    formula = "".join(
        [
            str(randint(1, max_range)),
            operators[randint(0, 3)],
            str(randint(1, max_range)),
        ]
    )

    return complete_eq(formula)


if __name__ == "__main__":
    for i in range(10):
        print(simple_eq())
