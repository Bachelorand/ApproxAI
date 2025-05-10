import tkinter as tk
import numpy as np
import csv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def clear_plot():
    for widget in plot_frame.winfo_children():
        widget.destroy()

def train_and_predict():
    try:
        clear_plot()

        # 1) Formel aus GUI
        formula = formula_entry.get().strip()
        if not formula:
            raise ValueError("Bitte eine Funktion in x1 und x2 eingeben, z. B. x1**2 + x2")

        # 2) Anzahl Zufallsdaten
        rnd_n = int(random_count_entry.get().strip())
        if not (0 <= rnd_n <= 1000):
            raise ValueError("Zufalls-Daten bitte zwischen 0 und 1000 wählen.")

        # 3) Hidden-Neuronen validieren
        hidden_n = int(hidden_neurons_entry.get().strip())
        if not (0 <= hidden_n <= 100):
            raise ValueError("Hidden-Neuronen bitte zwischen 0 und 100 wählen.")

        # 4) Wertebereich einlesen
        min_range = float(min_range_entry.get().strip())
        max_range = float(max_range_entry.get().strip())
        if min_range >= max_range:
            raise ValueError("Minimum muss kleiner als Maximum sein.")

        # 5) Manuelle Trainingsdaten
        x_train, y_train = [], []
        manual_n = 0
        for e1, e2 in entries_train:
            s1, s2 = e1.get().strip(), e2.get().strip()
            if not s1 or not s2:
                continue
            manual_n += 1
            x1, x2 = float(s1), float(s2)
            y = eval(formula, {"__builtins__": None, "np": np, "x1": x1, "x2": x2}, {})
            sq_raw = x1**2
            sq = np.clip(sq_raw, min_range, max_range)
            x_train.append([x1, x2, sq])
            y_train.append(y)

        # 6) Zufallsdaten im vorgegebenen Wertebereich
        for x1, x2 in np.random.uniform(min_range, max_range, size=(rnd_n, 2)):
            y = eval(formula, {"__builtins__": None, "np": np, "x1": x1, "x2": x2}, {})
            sq_raw = x1**2
            sq = np.clip(sq_raw, min_range, max_range)
            x_train.append([x1, x2, sq])
            y_train.append(y)

        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        # 7) Optionaler CSV-Export
        if export_var.get():
            csv_path = 'trainingsdaten.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x1', 'x2', 'x1_sq', 'y'])
                for xi, yi in zip(x_train, y_train):
                    writer.writerow([xi[0], xi[1], xi[2], yi])
            export_info = f"Trainingsdaten exportiert: {csv_path}\n\n"
        else:
            export_info = ""

        # 8) Testdaten
        x_test, y_test = [], []
        for idx, (t1, t2) in enumerate(entries_test):
            s1, s2 = t1.get().strip(), t2.get().strip()
            if not s1 or not s2:
                raise ValueError(f"Bitte beide Werte für Test-Paar {idx+1} eingeben.")
            a, b = float(s1), float(s2)
            true = eval(formula, {"__builtins__": None, "np": np, "x1": a, "x2": b}, {})
            sq_raw = a**2
            sq = np.clip(sq_raw, min_range, max_range)
            x_test.append([a, b, sq])
            y_test.append(true)

        x_test = np.array(x_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)

        # 9) Skalierung jedes Features in [min_range, max_range]
        X_min = x_train.min(axis=0)
        X_max = x_train.max(axis=0)
        denom = np.where((X_max - X_min) == 0, 1, (X_max - X_min))
        scale = max_range - min_range
        x_train_scaled = (x_train - X_min) / denom * scale + min_range
        x_test_scaled  = (x_test  - X_min) / denom * scale + min_range

        # 10) Modell mit RMSE
        model = Sequential([
            Dense(hidden_n, input_dim=3, activation='relu'),
            Dense(max(4, hidden_n//2), activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=[RootMeanSquaredError()]
        )

        # 11) Training
        history = model.fit(
            x_train_scaled, y_train,
            epochs=500,
            validation_split=0.2,
            callbacks=[ReduceLROnPlateau(patience=10, factor=0.5)],
            verbose=0
        )

        # 12) Vorhersage & manuelle Metriken
        preds = model.predict(x_test_scaled).flatten()
        mse  = float(np.mean((y_test - preds)**2))
        mae  = float(np.mean(np.abs(y_test - preds)))
        ss_res = np.sum((y_test - preds)**2)
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        r2   = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

        # 13) Ergebnisse anzeigen
        output = (
            f"Funktion: {formula}\n"
            f"Wertebereich (Train): [{min_range}, {max_range}]\n"
            f"Manuelle Paare: {manual_n}\n"
            f"Zufällige Paare: {rnd_n}\n"
            f"Trainings-Beispiele insgesamt: {len(x_train)}\n\n"
            + export_info +
            f"Test-Metriken → MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}\n\n"
            "Testdaten → Vorhersage:\n"
        )
        for (a, b, _), true, pred in zip(x_test.tolist(), y_test, preds):
            expr = formula.replace('x1', str(a)).replace('x2', str(b))
            output += f"{expr} = {pred:.2f}  (Soll: {true:.2f})\n"

        results_display.config(state='normal')
        results_display.delete('1.0', tk.END)
        results_display.insert(tk.END, output)
        results_display.config(state='disabled')

        # 14) Loss Function Diagram
        fig = plt.Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(history.history['root_mean_squared_error'], label='Train RMSE')
        ax.plot(history.history['val_root_mean_squared_error'], label='Val   RMSE')
        ax.set_title('Loss Function')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        canvas_plot = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_plot.draw()
        canvas_plot.get_tk_widget().pack(side=tk.LEFT, padx=10, expand=True)

    except Exception as err:
        results_display.config(state='normal')
        results_display.delete('1.0', tk.END)
        results_display.insert(tk.END, f"Fehler: {err}")
        results_display.config(state='disabled')


# --- GUI-Aufbau ---
root = tk.Tk()
root.title("ApproxAI")

canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas.pack(side="left", fill="both", expand=True)

frame = tk.Frame(canvas)
frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0,0), window=frame, anchor="nw")
canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

row = 0
# 1) Formel
tk.Label(frame, text="Funktion in x1, x2:").grid(row=row, column=0, columnspan=2, pady=5)
formula_entry = tk.Entry(frame, width=30)
formula_entry.insert(0, "x1**2 + x2")
formula_entry.grid(row=row+1, column=0, columnspan=2, pady=5)

# 2) Zufallsdaten
tk.Label(frame, text="Zufällige Paare (0–1000):").grid(row=row+2, column=0, pady=5)
random_count_entry = tk.Entry(frame, width=10)
random_count_entry.insert(0, "1000")
random_count_entry.grid(row=row+2, column=1, pady=5)

# 3) Manuelle Paare
entries_train = []
for i in range(3):
    e1 = tk.Entry(frame, width=10)
    e2 = tk.Entry(frame, width=10)
    e1.grid(row=row+3+i, column=0, padx=5, pady=2)
    e2.grid(row=row+3+i, column=1, padx=5, pady=2)
    entries_train.append((e1, e2))

# 4) Wertebereich
min_max_row = row+6
tk.Label(frame, text="Wertebereich Min:").grid(row=min_max_row, column=0, pady=5)
min_range_entry = tk.Entry(frame, width=10)
min_range_entry.insert(0, "-1")
min_range_entry.grid(row=min_max_row, column=1, pady=5)
tk.Label(frame, text="Wertebereich Max:").grid(row=min_max_row+1, column=0, pady=5)
max_range_entry = tk.Entry(frame, width=10)
max_range_entry.insert(0, "1")
max_range_entry.grid(row=min_max_row+1, column=1, pady=5)

# 5) Export-Option
export_var = tk.BooleanVar(value=False)
tk.Checkbutton(frame, text="CSV-Export aktivieren", variable=export_var) \
    .grid(row=min_max_row+2, column=0, columnspan=2, pady=5)

# 6) Testdaten
test_row = min_max_row+3
tk.Label(frame, text="Testdaten (x1, x2):").grid(row=test_row, column=0, columnspan=2, pady=10)
entries_test = []
for i in range(3):
    t1 = tk.Entry(frame, width=10)
    t2 = tk.Entry(frame, width=10)
    t1.grid(row=test_row+1+i, column=0, padx=5, pady=2)
    t2.grid(row=test_row+1+i, column=1, padx=5, pady=2)
    entries_test.append((t1, t2))

# 7) Hidden-Neuronen
hidden_row = test_row+4
tk.Label(frame, text="Hidden-Neuronen (0–100):").grid(row=hidden_row, column=0, columnspan=2, pady=10)
hidden_neurons_entry = tk.Entry(frame, width=10)
hidden_neurons_entry.insert(0, "100")
hidden_neurons_entry.grid(row=hidden_row+1, column=0, columnspan=2, pady=5)

# 8) Train-Button
tk.Button(frame, text="Trainieren & Vorhersagen", command=train_and_predict) \
    .grid(row=hidden_row+2, column=0, columnspan=2, pady=20)

# 9) Ergebnisse & Plot
res_row = hidden_row+3
tk.Label(frame, text="Ergebnisse:").grid(row=res_row, column=0, columnspan=2, pady=5)
results_display = tk.Text(frame, height=10, width=80, state='disabled')
results_display.grid(row=res_row+1, column=0, columnspan=2, padx=10, pady=10)

plot_row = res_row+2
tk.Label(frame, text="Loss Function:").grid(row=plot_row, column=0, columnspan=2)
plot_frame = tk.Frame(frame)
plot_frame.grid(row=plot_row+1, column=0, columnspan=2, pady=10)

# 10) Footer
footer = tk.Label(frame,
    text="© 2025 Yunus Hamurcu. Bachelorarbeit betreut von Prof. Dr. Ing. Ralf Otte, THU Ulm.",
    font=("Arial", 8)
)
footer.grid(row=plot_row+2, column=0, columnspan=2, pady=(5,10))

root.mainloop()
