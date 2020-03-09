# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause


def run_playground():
    import os
    import onelearn

    filename = onelearn.__file__.replace("__init__.py", "playground_decision.py")
    os.system("streamlit run %s" % filename)
