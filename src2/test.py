from callbacks import get_forward_outputs

outputs_path = "/Users/eitanturok/good-vibrations/runs/1777531033-pragmatic-bulldog/runs/1777531033-pragmatic-bulldog/forward_outputs"
rows = get_forward_outputs(outputs_path)
row = rows[0]
print(rows)
