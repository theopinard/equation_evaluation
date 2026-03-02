from random import randint

operators = "+-*/"


def complete_eq(formula: str) -> str:
    """
    evaluate the formula
    :param formula:
    :return:
    """
    result = str(round(eval(formula), 5))
    return "".join([formula, "=", result, "\n"])


def simple_eq() -> str:
    """
    generate simple equation with number in the
    :return:
    """
    formula = "".join(
        [str(randint(0, 9)), operators[randint(0, 3)], str(randint(0, 9)),]
    )
    try:
        result = complete_eq(formula)
    except ZeroDivisionError:
        result = simple_eq()
    return result


if __name__ == "__main__":
    for i in range(10):
        print(repr(simple_eq()))
