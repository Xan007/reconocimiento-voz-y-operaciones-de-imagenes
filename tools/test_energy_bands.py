import matplotlib.pyplot as plt
import json

uno = json.load(open("data/modelos/uno.json"))
dos = json.load(open("data/modelos/dos.json"))

plt.plot(uno["mean_energy"], label="uno", marker="o")
plt.plot(dos["mean_energy"], label="dos", marker="o")
plt.xlabel("Subbanda (0–15)")
plt.ylabel("Fracción de energía")
plt.title("Comparación de energía promedio por subbanda")
plt.legend()
plt.show()
