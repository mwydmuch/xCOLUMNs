every = 13
it = 5
data = [
{
    "start_row": 0,
    "columns": [1, 2, 13, 14]
},
{
    "start_row": 5,
    "columns": [5, 6, 17, 18]
},
{
    "start_row": 6,
    "columns": [5, 6, 17, 18]
},
{
    "start_row": 7,
    "columns": [7, 8, 19, 20]
},
{
    "start_row": 8,
    "columns": [7, 8, 19, 20]
},
{
    "start_row": 9,
    "columns": [9, 10, 21, 22]
},
{
    "start_row": 10,
    "columns": [9, 10, 21, 22]
},
{
    "start_row": 11,
    "columns": [11, 12, 23, 24]
},
{
    "start_row": 12,
    "columns": [11, 12, 23, 24]
},
]


every = 7
it = 6
data = [
{
    "start_row": 0,
    "columns": [1, 2, 11, 12, 21, 22]
},
{
    "start_row": 3,
    "columns": [5, 6, 15, 16, 25, 26]
},
{
    "start_row": 4,
    "columns": [7, 8, 17, 18, 27, 28]
},
{
    "start_row": 5,
    "columns": [7, 8, 17, 18, 27, 28]
},
{
    "start_row": 6,
    "columns": [9, 10, 19, 20, 29, 30]
},
]


every = 9
it = 6
data = [
{
    "start_row": 0,
    "columns": [3, 8, 13]
},
{
    "start_row": 1,
    "columns": [3, 8, 13]
},
{
    "start_row": 2,
    "columns": [3, 8, 13]
},
{
    "start_row": 3,
    "columns": [4, 9, 14]
},
{
    "start_row": 4,
    "columns": [4, 9, 14]
},
{
    "start_row": 5,
    "columns": [4, 9, 14]
},
{
    "start_row": 6,
    "columns": [5, 10, 15]
},
{
    "start_row": 7,
    "columns": [5, 10, 15]
},
{
    "start_row": 8,
    "columns": [5, 10, 15]
},
]

every = 6
it = 6
data = [
{
    "start_row": 0,
    "columns": [3, 8, 13]
},
{
    "start_row": 1,
    "columns": [3, 8, 13]
},
{
    "start_row": 2,
    "columns": [4, 9, 14]
},
{
    "start_row": 3,
    "columns": [4, 9, 14]
},
{
    "start_row": 4,
    "columns": [5, 10, 15]
},
{
    "start_row": 5,
    "columns": [5, 10, 15]
},
]


every = 8
it = 3
data = [
{
    "start_row": 0,
    "columns": [1, 8, 15]
},
{
    "start_row": 4,
    "columns": [3, 10, 17]
},
{
    "start_row": 5,
    "columns": [4, 11, 18]
},
{
    "start_row": 6,
    "columns": [5, 12, 19]
},
{
    "start_row": 7,
    "columns": [6, 13, 20]
},
]


# every = 7
# it = 5
# data = [
# {
#     "start_row": 1,
#     "columns": [3, 8, 13, 18]
# },
# {
#     "start_row": 2,
#     "columns": [3, 8, 13, 18]
# },
# {
#     "start_row": 3,
#     "columns": [4, 9, 14, 19]
# },
# {
#     "start_row": 4,
#     "columns": [4, 9, 14, 19]
# },
# {
#     "start_row": 5,
#     "columns": [5, 10, 15, 20]
# },
# {
#     "start_row": 6,
#     "columns": [5, 10, 15, 20]
# },
# ]

every = 7
it = 6
data = [
{
    "start_row": 0,
    "columns": [1, 8, 15]
},
{
    "start_row": 4,
    "columns": [3, 10, 17]
},
{
    "start_row": 5,
    "columns": [4, 11, 18]
},
{
    "start_row": 6,
    "columns": [5, 12, 19]
},
{
    "start_row": 7,
    "columns": [6, 13, 20]
},
]

every = 6
it = 3
data = [
{
    "start_row": 0,
    "columns": [1, 6]
},
{
    "start_row": 3,
    "columns": [3, 8]
},
{
    "start_row": 4,
    "columns": [4, 9]
},
{
    "start_row": 5,
    "columns": [5, 10]
},
]

every = 8
it = 5
data = [
{
    "start_row": 0,
    "columns": [1, 7, 13]
},
{
    "start_row": 4,
    "columns": [3, 9, 15]
},
{
    "start_row": 5,
    "columns": [4, 10, 16]
},
{
    "start_row": 6,
    "columns": [5, 11, 17]
},
{
    "start_row": 7,
    "columns": [6, 12, 18]
},
]

for d in data:
    start = d["start_row"]
    columns = d["columns"]
    for i in range(it):
        for c in columns:
            #print(f"every row no {start + i * every} column no {c}/.style={{postproc cell content/.append style={{/pgfplots/table/@cell content/.add={{\cellcolor{{Green!20!white}}}}{{}},}}}},")
            print(f"every row no {start + i * every} column no {c}/.style={{greencell}},")
    print("%")
