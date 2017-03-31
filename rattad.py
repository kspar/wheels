#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ast
from typing import List, Callable, Any
from functools import reduce
from grader import *


def msg_should_have_not_used_input():
    return "Selles ülesandes ei ole vaja midagi kasutaja käest küsida, seega ei tohiks 'input()' funktsiooni kasutada."


def msg_should_have_used_input():
    return "Selles ülesandes on vaja kasutaja käest andmeid küsida, selleks tuleb 'input()' funktsiooni kasutada."


def msg_should_have_func_call_toplevel(func_name):
    return "Programmi välimisel tasemel tuleb funktsiooni '{func}' välja kutsuda.".format(func=func_name)


def msg_should_have_while():
    return "Selles ülesandes on vaja while-tsüklit kasutada."


def msg_should_have_for():
    return "Selles ülesandes on vaja for-tsüklit kasutada."


def msg_should_have_loop():
    return "Selles ülesandes on vaja tsüklit kasutada."


def msg_should_have_func_def(func_name):
    return "Programmis peaks olema defineeritud funktsioon '{func}'. Ei leidnud selle funktsiooni definitsiooni.".format(
        func=func_name)


def msg_should_not_have_more_than_one_def(func_name):
    return "Leidsin programmist mitu funktsiooni '{func}' definitsiooni. Funktsioone tuleb defineerida vaid üks kord!".format(
        func=func_name)


def msg_should_have_returned_none(func_name, args_repr, ret_repr):
    return "Andsin funktsioonile '{func_name}' argumendiks {args_repr} ja ootasin, et see ei ' \
           'tagastaks mitte midagi. Kuid funktsioon tagastas {ret_repr}".format(func_name=func_name,
                                                                                args_repr=args_repr, ret_repr=ret_repr)


def msg_should_not_have_returned_none(func_name):
    return "Funktsioon '{func_name}' ei tagastanud midagi. Kas kasutasid returni?".format(func_name=func_name)


def msg_wrong_return_type(func_name):
    # TODO: ütle, mis tüüpi väärtust oodati (ja vb ka, mis saadi)
    return "Funktsioon '{func_name}' tagastas vale tüüpi väärtuse.".format(func_name=func_name)


def msg_wrong_return_value(func_name, args_repr, exp_ret, act_ret):
    # TODO: pane tööle ka 0 argumendi korral
    return "Andsin funktsioonile '{func_name}' argumendiks {args_repr} ja ootasin, et see tagastaks " \
           "{exp_ret} kuid see tagastas {act_ret}".format(func_name=func_name, args_repr=args_repr, exp_ret=exp_ret,
                                                          act_ret=act_ret)


def msg_nonpure_function(func_name):
    return "Funktsiooni '{func_name}' programmist väljastpoolt välja kutsudes tagastas see vale väärtuse. Kui teised testid lähevad läbi, " \
           "siis võib probleem olla selles, et kasutate funktsioonis argumentide asemel globaalseid muutujaid.".format(func_name=func_name)


def msg_wrong_number_of_params(func_name, exp_no_of_params, act_no_of_params):
    return "Funktsioon '{func}' peaks olema defineeritud {exp} parameetriga, kuid praegu " \
           "leidsin definitsioonist {act} parameetrit.".format(func=func_name, exp=exp_no_of_params,
                                                               act=act_no_of_params)


def msg_inner_function_call_missing(outer_func, inner_func):
    return "Ei leidnud funktsiooni '{outer_func}' seest funktsiooni '{inner_func}' väljakutset.".format(
        outer_func=outer_func, inner_func=inner_func)


def msg_recursive_function_call_missing(func_name):
    return "Ei leidnud funktsiooni '{func_name}' seest rekursiivset pöördumist enda ('{func_name}') poole.".format(
        func_name=func_name)


def msg_missing_expected_string_from_out(exp_out, actual_out):
    return "Ei leidnud programmi väljundist oodatud vastust {exp_out} Programmi väljund oli {actual_out}".format(
        exp_out=exp_out, actual_out=actual_out)


def msg_found_unexpected_string_from_out(unexp_out, actual_out):
    return "Leidsin programmi väljundist ootamatu vastuse {unexp} Programmi väljund oli {actual}".format(
        unexp=unexp_out, actual=actual_out)


def msg_no_distinct_return_values_from_random(func_name, samples):
    # TODO: make tööle n > 0 argumendiga
    return "Rakendasin funktsiooni '{func_name}' {n} korda ja iga kord tagastati sama tulemus.".format(
        func_name=func_name, n=samples)


def msg_illegal_return_value_from_random(func_name, actual_ret):
    # TODO: make tööle n > 0 argumendiga
    return "Funktsioon '{func_name}' tagastas ebakorrektse väärtuse: {act_ret}".format(func_name=func_name,
                                                                                       act_ret=actual_ret)


### LOW-LEVEL

def assrt(condition, message):
    # TODO: replace raw 'assert' with 'assrt'
    assert condition, message


def name_in_ast(node, name):
    # TODO: implement independently
    return ast_contains_name(node, name)


def quote(x):
    # TODO: implement independently
    return quote_text_block(str(x)) + "\n"


def equal_simple(obj1, obj2):
    return obj1 == obj2


def equal_list_no_order(lst1, lst2):
    return sorted(lst1) == sorted(lst2)


def equal_strings_strip(str1, str2):
    return str1.strip() == str2.strip()


def matrix_repr(mat):
    # TODO: could align columns to make it look even nicer
    return quote("[" + ",\n ".join(map(lambda r: str(r), mat)) + "]")


def quotemark_repr(strng):
    return '"{}"'.format(strng)


def at_least_2_unique(lst):
    for i, val1 in enumerate(lst):
        for val2 in lst[i:]:
            if val1 != val2:
                return True
    return False


def ast_contains_function_call(node, callee_name):
    for node in ast.walk(node):
        if isinstance(node, ast.Call) and hasattr(node, 'func') and hasattr(node.func, 'id') and node.func.id == callee_name:
            return True
    return False


### HIGH-LEVEL

def must_not_have_input(node: ast.AST):
    """
    Check that subtree does NOT contain calls to function 'input'.
    :param node:  root of subtree
    """
    # TODO: this should not allow calls, but should allow other Names
    assrt(not name_in_ast(node, "input"), msg_should_have_not_used_input())


def must_have_input(node: ast.AST):
    """
    Check subtree for calls to function 'input'.
    :param node: root of subtree
    """
    assrt(name_in_ast(node, "input"), msg_should_have_used_input())


def must_have_function_call(node: ast.AST, function_name: str):
    """
    Check subtree for nodes that represent a call to function_name. 
    :param node: root of subtree
    :param function_name:
    """
    assrt(ast_contains_function_call(node, function_name), msg_should_have_func_call_toplevel(function_name))


def must_have_nested_function_call(outer_func_def_node: ast.FunctionDef, inner_func_name: str):
    """
    Check that outer_func_def_node's subtree contains a call to inner_func_name. 
    :param outer_func_def_node: root of subtree
    :param inner_func_name: name of function that must be called 
    """
    assrt(ast_contains_function_call(outer_func_def_node, inner_func_name),
          msg_inner_function_call_missing(outer_func_def_node.name, inner_func_name))


def must_have_recursive_function_call(outer_func_def_node: ast.FunctionDef, func_name: str):
    """
    Check that outer_func_def_node's subtree contains a recursive call to itself.
    :param outer_func_def_node: root of subtree
    :param func_name: name of the (this) recursive function
    """
    assrt(ast_contains_function_call(outer_func_def_node, func_name),
          msg_recursive_function_call_missing(func_name))


def must_have_loop_while(node: ast.AST):
    """
    Check subtree for nodes that represent a while-loop.
    :param node: root of subtree 
    """
    assrt(ast_contains(node, ast.While), msg_should_have_while())


def must_have_loop_for(node: ast.AST):
    """
    Check subtree for nodes that represent a for-loop.
    :param node: root of subtree
    """
    # TODO: consider list comprehension
    assrt(ast_contains(node, ast.For), msg_should_have_for())


def must_have_loop(node: ast.AST):
    """
    Check subtree for nodes that represent any loop.
    :param node: root of subtree 
    """
    assrt(ast_contains(node, ast.While) or ast_contains(node, ast.For), msg_should_have_loop())


def write_dummy_data(stdin, stdout, dummy: List[str] = ['1337'] * 10):
    """
    Write dummy data to stdin. This might be needed to stop the student program from blocking or for other reasons.
    :param stdin: 
    :param stdout: 
    :param dummy: list of strings that are written to stdin
    """
    for d in dummy[:-1]:
        stdin.write(d)
    stdout.reset()
    stdin.write(dummy[-1])


def must_have_func_def_toplevel(module_, function_name: str):
    """
    Check that there is a toplevel def of function_name.
    :param module_: python-grader's m.module object
    :param function_name:
    """
    # TODO: add non-top-level checking as a separate function
    # TODO: implement using AST inspection
    assrt(hasattr(module_, function_name), msg_should_have_func_def(function_name))


def get_function(module_, function_name: str) -> Callable:
    """
    Get Python function object named function_name. Currently this function must be defined on top level.
    :param module_: python-grader's m.module object 
    :param function_name: 
    :return: corresponding Python function object
    """
    must_have_func_def_toplevel(module_, function_name)
    return getattr(module_, function_name)


def get_function_def_node(node: ast.AST, function_name: str) -> ast.FunctionDef:
    """
    Fetch function def AST node of function_name from subtree with root node.
    assrt()s that that there is exactly 1 definition.
    :param node: root of subtree
    :param function_name: 
    :return: corresponding function def AST node
    """
    defs = []
    for node in ast.walk(node):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            defs.append(node)
    assrt(len(defs) <= 1, msg_should_not_have_more_than_one_def(function_name))
    assrt(len(defs) >= 1, msg_should_have_func_def(function_name))
    return defs[0]


def must_have_n_params(func_def_node: ast.FunctionDef, no_of_params: int):
    """
    Checks that func_def_node is defined with exactly no_of_params params. 
    :param func_def_node: AST node of function def
    :param no_of_params: number of parameters
    """
    assrt(len(func_def_node.args.args) == no_of_params,
          msg_wrong_number_of_params(func_def_node.name, no_of_params, len(func_def_node.args.args)))


def must_have_equal_return_values(expected_func_object: Callable, actual_func_object: Callable, function_name: str, *func_args: Any,
                                  equalizer: Callable = equal_simple, ret_representer: Callable = quote, args_repr: str = None):
    """
    Check if two given Python function objects have equal return values when provided with *func_args.
    
    A custom equalizer can be provided for return value equality check (default is ==). The equalizer must
    have 2 params and return a bool.
    A custom representer can be provided for displaying the (differing) return values. The representer should
    return a string.
    A custom representation of given arguments can be provided as a string. Note that all arguments must be 
    included in this representation. This can be used for argument types that don't repr() well, i.e. matrices.  
    
    :param expected_func_object: 
    :param actual_func_object: 
    :param function_name: 
    :param func_args: 
    :param equalizer: 
    :param ret_representer: 
    :param args_repr: 
    """
    # TODO: should have a list of args instead of *
    # TODO: handle funcs with no params
    expected_return = expected_func_object(*func_args)
    actual_return = actual_func_object(*func_args)

    if args_repr is None:
        # TODO: should this use repr or str?
        args_repr = reduce(lambda a1, a2: repr(a1) + ', ' + repr(a2), func_args)

    if expected_return is None:
        # we expect the student function to 'not return anything'
        assert actual_return is None, msg_should_have_returned_none(function_name, args_repr, ret_representer(actual_return))
    else:
        assert actual_return is not None, msg_should_not_have_returned_none(function_name)
        # should this be (actual_return == expected_return) == (str(..) == str(..)) ?
        assert isinstance(actual_return, expected_return.__class__), msg_wrong_return_type(function_name)
        assert equalizer(actual_return, expected_return), \
            msg_wrong_return_value(function_name, args_repr, ret_representer(expected_return),
                                   ret_representer(actual_return))


def must_have_pure_func(expected_func_object: Callable, actual_func_object: Callable, function_name: str,
                        real_args: List[Any], mock_args: List[Any], equalizer=equal_simple, ret_representer=quote, args_repr=None):
    """
    Use real and mock arguments to check if a given student function is pure (referentially transparent).
    The same mock args should be provided to the student program beforehand.
    The student function actual_func_object is considered to be nonpure iff
    equalizer(actual_func_object(*real_args), expected_func_object(*mock_args)).
    Otherwise the student function is considered as correct or incorrect based on its return value.
    NB! expected_func_object(*real_args) must NOT return None!
    
    :param expected_func_object: 
    :param actual_func_object: 
    :param function_name: 
    :param real_args: 
    :param mock_args: 
    :param equalizer: see must_have_equal_return_values
    :param ret_representer: see must_have_equal_return_values
    :param args_repr: see must_have_equal_return_values
    """
    expected_return = expected_func_object(*real_args)
    actual_return = actual_func_object(*real_args)
    expected_mock_return = expected_func_object(*mock_args)

    if args_repr is None:
        # TODO: should this use repr or str?
        args_repr = reduce(lambda a1, a2: repr(a1) + ', ' + repr(a2), real_args)

    assert expected_return is not None, 'Solution function must not return None'

    # Student function should not return None
    assrt(actual_return is not None, msg_should_not_have_returned_none(function_name))
    # Check if student function is nonpure
    assrt(not equalizer(actual_return, expected_mock_return), msg_nonpure_function(function_name))
    # Check if student function's return type is correct
    assrt(isinstance(actual_return, expected_return.__class__), msg_wrong_return_type(function_name))
    # Check if student function's return value is correct
    assrt(equalizer(actual_return, expected_return), msg_wrong_return_value(function_name, args_repr,
                                                                            ret_representer(expected_return), ret_representer(actual_return)))


def must_have_correct_random_return_value(actual_func_object: Callable, function_name: str, func_args: List[Any],
                                          validation_function: Callable, uniqueness_function: Callable = at_least_2_unique, samples: int = 100):
    """
    Run the given function samples times. Check if each return value is valid using validation_function and
    if the return values are 'random enough' using uniqueness_function.
    The validation_function will be called with 1 argument - return value of the student function - and is expected to 
    return a bool representing whether the given return value is valid.
    A custom uniqueness_function with 1 parameter and bool return value can be provided. This function will be called
    with a list of all return values of the student function and is expected to return a bool whether the values are
    'unique enough' (default requires at least 2 distinct values).
    :param actual_func_object: 
    :param function_name: 
    :param func_args: 
    :param validation_function: function with 1 parameter and bool return value
    :param uniqueness_function: 
    :param samples: 
    """
    return_values = []
    for _ in range(samples):
        actual_return = actual_func_object(*func_args)
        assrt(validation_function(actual_return), msg_illegal_return_value_from_random(function_name, actual_return))
        return_values.append(actual_return)
    # TODO: message should say that values must be 'more unique' since must take into account any uniqueness_function?
    assrt(uniqueness_function(return_values), msg_no_distinct_return_values_from_random(function_name, samples))


def must_have_correct_output_str(stdin, stdout, inputs: List[Any], expected_strings: List[str], unexpected_strings: List[str],
                                 exp_out_representer: Callable = quote, act_out_representer: Callable = quote, ignore_case: bool = True):
    """
    Write inputs to stdin. Read stdout. Check if all strings in expected_strings are contained in out.
    Check if none of the strings in unexpected_strings are contained in out.
    :param stdin: 
    :param stdout: 
    :param inputs: 
    :param expected_strings: 
    :param unexpected_strings: 
    :param exp_out_representer: 
    :param act_out_representer: 
    :param ignore_case: 
    """
    # write all but the last input
    if len(inputs) > 1:
        for inp in inputs[:-1]:
            stdin.write(inp)

    if len(inputs) > 0:
        # if there was inputs, reset stdout and write last one since others were written already
        stdout.reset()
        stdin.write(inputs[-1])

    out = stdout.new()
    for exp in expected_strings:
        if ignore_case:
            assert exp.upper() in out.upper(), msg_missing_expected_string_from_out(exp_out_representer(exp),
                                                                                    act_out_representer(out))
        else:
            assert exp in out, msg_missing_expected_string_from_out(exp_out_representer(exp), act_out_representer(out))
    for unexp in unexpected_strings:
        if ignore_case:
            assert unexp.upper() not in out.upper(), msg_found_unexpected_string_from_out(exp_out_representer(unexp),
                                                                                          act_out_representer(out))
        else:
            assert unexp not in out, msg_found_unexpected_string_from_out(exp_out_representer(unexp),
                                                                          act_out_representer(out))


def write_inputs(stdin, stdout, inputs: List[str]):
    """
    Write inputs to stdin. Reset output stream after writing.
    :param stdin: 
    :param stdout: 
    :param inputs: 
    """
    # write all but the last input
    if len(inputs) > 1:
        for inp in inputs[:-1]:
            stdin.write(inp)

    if len(inputs) > 0:
        # if there was inputs, reset stdout and write last one since others were written already
        stdout.reset()
        stdin.write(inputs[-1])
