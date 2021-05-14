from docstring_parser import parse
import re
import os
import inspect
import importlib

import csdl

lang_pkg = csdl

# choose package that implements CSDL
import csdl_om

impl_pkg = csdl_om

# set paths
lang_example_class_definition_directory = inspect.getfile(
    lang_pkg)[:-len('__init__.py')] + 'examples/'
impl_example_class_definition_directory = inspect.getfile(
    impl_pkg)[:-len('__init__.py')] + 'examples/'
lang_test_exceptions_directory = \
     lang_example_class_definition_directory + 'invalid/'
impl_test_exceptions_directory = \
     impl_example_class_definition_directory + 'invalid/'
lang_test_computations_directory = \
    lang_example_class_definition_directory + 'valid/'
impl_test_computations_directory = \
     impl_example_class_definition_directory + 'valid/'


def camel_to_snake(name):
    return re.sub(
        '([a-z0-9])([A-Z])',
        r'\1_\2',
        re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name),
    ).lower()


def write_run_phase(f, obj, options):
    # create simulator from model
    f.write('sim = Simulator(' + obj.__name__ + '(')

    if len(options) > 0:
        f.write('\n')
    for opt in options:
        f.write('    ' + opt + ',\n')
    f.write('))\n')

    # run simulation
    f.write('sim.run()\n')


def get_example_file_name(obj, py_file_path):
    obj_name_snake_case = camel_to_snake(obj.__name__)
    prefix = ''
    example_filename = None
    if obj_name_snake_case[:len('error_')] == 'error_':
        prefix = 'error_'
        example_filename = py_file_path.rsplit(
            '/', 1
        )[-1][:-len('.py')] + '_' + obj_name_snake_case[len(prefix):] + '.py'
    if obj_name_snake_case[:len('example_')] == 'example_':
        prefix = 'example_'
        example_filename = py_file_path.rsplit(
            '/', 1
        )[-1][:-len('.py')] + '_' + obj_name_snake_case[len(prefix):] + '.py'
    return example_filename, prefix


def export_examples(
    pkg_with_example_class_definitions,
    output_directory_example_exceptions,
    output_directory_example_computations,
):
    # Python 3.9: use removesuffix
    example_class_definition_module = pkg_with_example_class_definitions.__name__ + '.examples'
    example_class_definition_directory = inspect.getfile(
        pkg_with_example_class_definitions)[:-len('__init__.py')] + 'examples/'
    for examples_file in os.listdir(example_class_definition_directory):
        suffix = '.py'
        if examples_file[-len(suffix):] == suffix:
            example_classes_file_path = (example_class_definition_directory +
                                         examples_file)

            # gather imports
            import_statements = []
            with open(example_classes_file_path, 'r') as f:
                import_statements = []
                for line in f:
                    line_text = line.lstrip()
                    if re.match('import', line_text) or re.match(
                            'from', line_text):
                        import_statements.append(line_text)

            # Python 3.9: use removesuffix
            lang_examples_module = importlib.import_module(
                example_class_definition_module + '.' +
                examples_file[:-len(suffix)])
            members = inspect.getmembers(lang_examples_module)
            for obj in dict(members).values():
                if inspect.isclass(obj):
                    print(
                        'Generating example script for class {} from file {}'.
                        format(obj.__name__, examples_file))
                    example_run_file_name, prefix = \
                        get_example_file_name(
                            obj, example_classes_file_path,
                        )

                    if example_run_file_name is not None:
                        # collect params
                        docstring = parse(obj.__doc__)
                        var_names = []
                        options = []
                        for param in docstring.params:
                            if param.arg_name == 'var':
                                var_names.append(param.description)
                            if param.arg_name == 'option':
                                options.append(param.description)

                        example_run_file_path = None
                        if prefix == 'error_':
                            example_run_file_path = \
                                output_directory_example_exceptions \
                                + example_run_file_name
                        elif prefix == 'example_':
                            example_run_file_path = \
                                output_directory_example_computations \
                                + example_run_file_name
                        print('writing to file', example_run_file_path)
                        with open(example_run_file_path, 'w') as f:
                            # write import statements
                            for stmt in import_statements:
                                f.write(stmt)
                            f.write('from csdl_om import Simulator\n')
                            f.write('\n\n')

                            # write example class
                            source = re.sub('.*:param.*:.*\n', '',
                                            inspect.getsource(obj))
                            source = re.sub('\n.*"""\n.*"""', '', source)
                            f.write(source)
                            f.write('\n\n')

                            write_run_phase(f, obj, options)

                            # output values
                            if len(var_names) > 0:
                                f.write('\n')
                            for var in var_names:
                                f.write('print(\'' + var + '\', sim[\'' + var +
                                        '\'].shape)\n')
                                f.write('print(sim[\'' + var + '\'])')
                                f.write('\n')
                            f.close()


# export_examples(
#     lang_pkg,
#     lang_test_exceptions_directory,
#     lang_test_computations_directory,
# )

# generate run scripts from examples in CSDL package using this
# implementation of CSDL
print('START: implementation-agnostic examples')
export_examples(
    lang_pkg,
    impl_test_exceptions_directory,
    impl_test_computations_directory,
)
print('END: implementation-agnostic examples')
# generate run scripts from examples specific to this implementation of
# CSDL
print('START: implementation-specific examples')
export_examples(
    impl_pkg,
    impl_test_exceptions_directory,
    impl_test_computations_directory,
)
print('END: implementation-specific examples')
