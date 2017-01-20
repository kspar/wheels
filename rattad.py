#!/usr/bin/python3
# -*- coding: utf-8 -*-

import ast
from functools import reduce
from grader import *


def msg_should_have_not_used_input():
    return "Selles ülesandes ei ole vaja midagi kasutaja käest küsida, seega ei tohiks 'input()' funktsiooni kasutada."


def msg_should_have_used_input():
    return "Selles ülesandes on vaja kasutaja käest andmeid küsida, selleks tuleb 'input()' funktsiooni kasutada."


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


### HIGH-LEVEL

def must_not_have_input(node):
    # TODO: this should not allow calls, but should allow other names
    assert not name_in_ast(node, "input"), msg_should_have_not_used_input()


def must_have_input(node):
    assert name_in_ast(node, "input"), msg_should_have_used_input()


def must_have_nested_function_call(outer_func_def_node, inner_func_name):
    assert ast_contains_function_call(outer_func_def_node, inner_func_name), \
        msg_inner_function_call_missing(outer_func_def_node.name, inner_func_name)


def must_have_recursive_function_call(outer_func_def_node, func_name):
    assert ast_contains_function_call(outer_func_def_node, func_name), \
        msg_recursive_function_call_missing(func_name)


def ast_contains_function_call(node, callee_name):
    for node in ast.walk(node):
        if isinstance(node, ast.Call) and hasattr(node, 'func') and hasattr(node.func, 'id') and node.func.id == callee_name:
            return True
    return False


def write_dummy_data(stdin, stdout, dummy=['1337']*10): 
    for d in dummy[:-1]:
        stdin.write(d)
    stdout.reset()
    stdin.write(dummy[-1])


def must_have_func_def_toplevel(module, fname):
    # TODO: add non-top-level checking as a separate function

    assert hasattr(module, fname), msg_should_have_func_def(fname)


def get_function(module, fname):
    must_have_func_def_toplevel(module, fname)
    return getattr(module, fname)


def get_function_def_node(node, fname):
    defs = []
    for node in ast.walk(node):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            defs.append(node)
    assert len(defs) <= 1, msg_should_not_have_more_than_one_def(fname)
    assert len(defs) >= 1, msg_should_have_func_def(fname)
    return defs[0]


def must_have_n_params(func_def_node, no_of_args):
    assert len(func_def_node.args.args) == no_of_args, \
        msg_wrong_number_of_params(func_def_node.name, no_of_args, len(func_def_node.args.args))


def must_have_equal_return_values(expected_func_object, actual_func_object, function_name, *func_args,
                                  equalizer=equal_simple, ret_representer=quote, args_repr=None):
    # TODO: add checking for global vars change
    # TODO: should have a list of args instead of *
    # TODO: handle funcs with no params
    expected_return = expected_func_object(*func_args)
    actual_return = actual_func_object(*func_args)

    if args_repr is None:
        # TODO: should this use repr or str?
        args_repr = reduce(lambda a1, a2: repr(a1) + ', ' + repr(a2), func_args)

    if expected_return is None:
        # we expect the student function to 'not return anything'
        assert actual_return is None, msg_should_have_returned_none(function_name, args_repr,
                                                                    ret_representer(actual_return))
    else:
        assert actual_return is not None, msg_should_not_have_returned_none(function_name)
        # should this be (actual_return == expected_return) == (str(..) == str(..)) ?
        assert isinstance(actual_return, expected_return.__class__), msg_wrong_return_type(function_name)
        assert equalizer(actual_return, expected_return), \
            msg_wrong_return_value(function_name, args_repr, ret_representer(expected_return),
                                   ret_representer(actual_return))


def must_have_correct_random_return_value(actual_func_object, function_name, func_args,
                                          validation_function, uniqueness_function=at_least_2_unique, samples=100):
    # TODO: add checking for global vars change
    return_values = []
    for _ in range(samples):
        actual_return = actual_func_object(*func_args)
        assrt(validation_function(actual_return), msg_illegal_return_value_from_random(function_name, actual_return))
        return_values.append(actual_return)
    # TODO: message should say that values must be 'more unique' since must take into account any uniqueness_function?
    assrt(uniqueness_function(return_values), msg_no_distinct_return_values_from_random(function_name, samples))


def must_have_correct_output_str(stdin, stdout, inputs, expected_strings, unexpected_strings,
                                 exp_out_representer=quote, act_out_representer=quote, ignore_case=True):
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
