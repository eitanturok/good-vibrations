from callbacks import get_forward_outputs

outputs_path = "/Users/eitanturok/good-vibrations/runs/1777503259-eggplant-pug/runs/1777503259-eggplant-pug/forward_outputs"
rows = get_forward_outputs(outputs_path)
row = rows[0]
print(len(rows))
