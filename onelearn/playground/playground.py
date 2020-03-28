# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause


def run_playground_decision():
    import os
    import onelearn

    filename = onelearn.__file__.replace(
        "/onelearn/__init__.py", "/examples/playground_decision.py"
    )
    os.system("streamlit run %s" % filename)


def run_playground_tree():
    import os
    import onelearn

    filename = onelearn.__file__.replace(
        "/onelearn/__init__.py", "/examples/playground_tree.py"
    )
    os.system("streamlit run %s" % filename)
