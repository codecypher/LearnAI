#!/usr/bin/env python3
"""
mindmap.py
This program creates a mindmap using Python and Graphviz.

Requirements: graphviz, pydot

[Make a Mindmap with Python and Graphviz](https://medium.com/analytics-vidhya/make-a-mindmap-with-python-and-graphviz-7aee20a3a9a3)
"""
import os
import pydot


def main():
    file_name = "curriculum.png"

    # Delete file if it exists
    if os.path.exists(file_name):
        os.remove(file_name)

    graph = pydot.Dot(graph_type="digraph", rankdir="LR")

    root = "Curriculum"
    days = ["Mar 1", "Mar 2", "Mar 3", "Mar 4", "Mar 5", "Mar 6"]
    for day in days:
        graph.add_edge(pydot.Edge(root, day))

    day1 = [
        "Basicy Python Syntax",
        "Modularization - Function",
        "Modularization - Class",
        "Modularization - Package",
        "Modularization - Study Case",
    ]

    for d1 in day1:
        graph.add_edge(pydot.Edge(days[0], d1))

    day1_basic = ["Sequential Code", "Branching with 1F", "Looping with For and While"]

    for day2 in day1_basic:
        graph.add_edge(pydot.Edge(day1[0], day2))

    day1_class = ["Polymorphism", "Inheritance", "Encapsulation"]

    for day2 in day1_class:
        graph.add_edge(pydot.Edge(day1[2], day2))

    graph.write_png(file_name)


# Check that code is under main function
if __name__ == "__main__":
    main()
    print("\nDone!")
