import torch
from rich import print
from typing import List


def pretty_print_mat(
    mat: torch.Tensor, column_labels: List[str], row_labels: List[str], highlight=True
):
    print("".join([" " * 20] + [f"{i:>5.5} | " for i in [""] + column_labels]))
    for i in range(len(row_labels)):
        print(f"{row_labels[i]:>25.25} | ", end="")
        for j in range(len(column_labels)):
            if highlight:
                if i == j:
                    print(f"[blue]{mat[i, j]:>5}[/blue] | ", end="")
                else:
                    if mat[i, j] == mat[i].max():
                        print(f"[red]{mat[i, j]:>5}[/red] | ", end="")
                    else:
                        print(f"[yellow]{mat[i, j]:>5}[/yellow] | ", end="")
            else:
                print(f"[yellow]{mat[i, j]:>5}[/yellow] | ", end="")
        print()
