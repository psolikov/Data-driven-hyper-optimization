# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import re
import ast
from collections import defaultdict


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


def add_missing_newlines(line: str):
    return line


def parse_data_item(item: str):
    check_beginning_re = re.compile('^Модель [1-9][0-9]*:')
    match = re.match(check_beginning_re, item)
    if not match:
        raise RuntimeError('Wrong data format. Should start with \"Модель <number>:\"')
    code_item = item[match.end() + 1:]
    code_item = add_missing_newlines(code_item)
    return ast.parse(code_item)


def get_model_and_data_names(code_ast: ast.Module):
    found = False
    model_name, X_name, y_name = None, None, None
    for item in code_ast.body:
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call) and item.value.func.attr == 'fit':
            if found:
                raise RuntimeError('Data error: multiple fit calls in one data point.')
            found = True
            model_name = item.value.func.value.id
            if len(item.value.args) != 2:
                raise RuntimeError('Data error: fit should have two args.')
            X_name = item.value.args[0].id
            y_name = item.value.args[1].id
    if not found:
        raise RuntimeError('Data error: no fit call.')
    return model_name, X_name, y_name


def get_model_arguments(code_ast: ast.Module, model_name: str):
    found = False
    model_class, model_args = None, None
    for item in code_ast.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and \
                item.targets[0].id == model_name:
            if found:
                raise RuntimeError('Data error: multiple model assignments in one data point.')
            found = True
            model_class = item.value.func.attr
            model_args = {}
            for keyword in item.value.keywords:
                arg, value = keyword.arg, keyword.value
                model_args[arg] = ast.literal_eval(value)
    if not found:
        raise RuntimeError('Data error: not found model creation.')
    return model_class, model_args


def get_data_args(code_ast):
    found = False
    data_args = {}
    for item in code_ast.body:
        if isinstance(item, ast.Assign) and isinstance(item.value,
                                                       ast.Call) and isinstance(item.value.func,
                                                                                ast.Name) and item.value.func.id == 'train_test_split':
            if found:
                raise RuntimeError('Data error: multiple model assignments in one data point.')
            found = True
            data_args = {}
            for keyword in item.value.keywords:
                arg, value = keyword.arg, keyword.value
                data_args[arg] = ast.literal_eval(value)
    return data_args


def make_arguments_vector(model_args, data_args):
    # todo find all arguments of the class with default values
    return {**model_args, **data_args}


def write_model_arguments_data(model_arguments_stats):
    pass


if __name__ == '__main__':
    data_file = open('data.txt', 'r')
    lines = []
    tmp_dict = [
        "Модель 1: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)\nxg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)\nxg_reg.fit(X_train,y_train)",
    "Модель 2: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)\nxg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.5, learning_rate = 0.0001, n_estimators = 10)\nxg_reg.fit(X_train,y_train)",
    "Модель 3: X = [[0, 0], [2, 2]]\ny = [0.5, 2.5]\nregr = svm.SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,coef0=1)\nregr.fit(X, y)"]
    model_arguments_data = defaultdict(list)

    for line in tmp_dict:
        # for line in data_file.readlines():
        code_ast = parse_data_item(line)
        model_name, X_name, y_name = get_model_and_data_names(code_ast)
        model_class, model_args = get_model_arguments(code_ast, model_name)
        data_args = get_data_args(code_ast)
        args_vector = make_arguments_vector(model_args, data_args)
        model_arguments_data[model_class].append(args_vector)

    write_model_arguments_data(model_arguments_data, file='')
    print_model_arguments_stats(model_arguments_data, model='')
