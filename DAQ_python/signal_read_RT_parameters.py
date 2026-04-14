import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time
from scipy.signal import find_peaks

# ================= KONFIGURACJA =================
PORT_SZEREGOWY = 'COM10'
PREDKOSC = 115200
MAX_PUNKTOW = 200

# Parametry ADC
V_REF = 5.0
ADC_RES = 1023.0
# ================================================

try:
    ser = serial.Serial(PORT_SZEREGOWY, PREDKOSC, timeout=0.1)
except Exception as e:
    print(f"Błąd połączenia: {e}")
    exit()

x_data = deque(maxlen=MAX_PUNKTOW)
y_data = deque(maxlen=MAX_PUNKTOW)
counter = 0
last_time = time.time()
fs_real = 0

fig, ax = plt.subplots(figsize=(11, 6))
plt.subplots_adjust(right=0.75) 
line, = ax.plot([], [], lw=2, color='#2ecc71', label='Sygnał')
peak_dots, = ax.plot([], [], 'ro', label='Piki') # Czerwone kropki na pikach

ax.set_title('Akwizycja Live - Detekcja Pików')
ax.set_xlabel('Numer próbki')
ax.set_ylabel('Napięcie [V]')
ax.grid(True, alpha=0.3)

stats_text = ax.text(1.05, 0.9, '', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def calculate_params(data, fs):
    if len(data) < 20: return "Analiza..."
    
    y = np.array(data)
    v_max = np.max(y)
    v_min = np.min(y)
    v_pp = v_max - v_min
    offset = np.mean(y)
    v_rms = np.sqrt(np.mean(y**2))
    
    # --- WYKRYWANIE PIKÓW DLA CZĘSTOTLIWOŚCI ---
    # distance: minimalna liczba próbek między pikami (zapobiega wykrywaniu szumu jako piki)
    # prominence: minimalna wysokość piku względem otoczenia
    peaks, _ = find_peaks(y, distance=fs/50 if fs > 50 else 2, prominence=v_pp*0.2)
    
    freq_sig = 0
    if len(peaks) > 1:
        # Obliczamy średni odstęp między pikami w próbkach
        avg_peak_dist = np.mean(np.diff(peaks))
        if avg_peak_dist > 0:
            freq_sig = fs / avg_peak_dist

    # Zwracamy statystyki oraz indeksy pików do wizualizacji
    stats = (f"PARAMETRY [V]:\n"
            f"------------------\n"
            f"V_max:  {v_max:>7.3f} V\n"
            f"V_min:  {v_min:>7.3f} V\n"
            f"V_pp:   {v_pp:>7.3f} V\n"
            f"Offset: {offset:>7.3f} V\n"
            f"V_rms:  {v_rms:>7.3f} V\n"
            f"------------------\n"
            f"F_sig:  {freq_sig:>7.2f} Hz\n"
            f"Fs_real:{fs:>7.1f} Hz")
    
    return stats, peaks

def update(frame):
    global counter, last_time, fs_real
    
    try:
        while ser.in_waiting > 0:
            raw_data = ser.readline().decode('utf-8').strip()
            if raw_data:
                now = time.time()
                dt = now - last_time
                if dt > 0:
                    fs_real = 1.0 / dt
                last_time = now

                adc_value = float(raw_data)
                voltage = (adc_value * V_REF) / ADC_RES
                
                x_data.append(counter)
                y_data.append(voltage)
                counter += 1
        
        if y_data:
            line.set_data(x_data, y_data)
            
            # Obliczenia i aktualizacja tekstu
            stats_str, peak_indices = calculate_params(list(y_data), fs_real)
            stats_text.set_text(stats_str)
            
            # Wizualizacja pików na wykresie
            if len(peak_indices) > 0:
                # Przeliczenie indeksów z bufora na wartości osi X
                peak_x = [list(x_data)[i] for i in peak_indices]
                peak_y = [list(y_data)[i] for i in peak_indices]
                peak_dots.set_data(peak_x, peak_y)
            else:
                peak_dots.set_data([], [])

            # Skalowanie osi
            ax.set_xlim(x_data[0], x_data[-1] + 1)
            min_y, max_y = min(y_data), max(y_data)
            margin = max(0.2, (max_y - min_y) * 0.1)
            ax.set_ylim(min_y - margin, max_y + margin)
            
    except Exception:
        pass
    return line, stats_text, peak_dots



ani = animation.FuncAnimation(fig, update, interval=30, blit=False)

print(f"Detekcja pików aktywna. V_REF={V_REF}V")
plt.show(block=True)
ser.close()