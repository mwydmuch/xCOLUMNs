# Script for generating highlights for latex table

# Highlights description:
every = 8
it = 5
data = [
    {"start_row": 0, "columns": [1, 7, 13]},
    {"start_row": 4, "columns": [3, 9, 15]},
    {"start_row": 5, "columns": [4, 10, 16]},
    {"start_row": 6, "columns": [5, 11, 17]},
    {"start_row": 7, "columns": [6, 12, 18]},
]

# Print highlights
for d in data:
    start = d["start_row"]
    columns = d["columns"]
    for i in range(it):
        for c in columns:
            print(
                f"every row no {start + i * every} column no {c}/.style={{greencell}},"
            )
    print("%")
