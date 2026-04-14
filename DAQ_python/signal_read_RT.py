import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time

# ================= KONFIGURACJA =================
PORT_SZEREGOWY = 'COM10'
PREDKOSC = 115200
MAX_PUNKTOW = 200

# Parametry ADC do przeliczenia na napięcie
V_REF = 5.0
ADC_RES = 1023.0
# ================================================

try:
    # Ustawiamy krótki timeout, aby readline nie blokowało animacji
    ser = serial.Serial(PORT_SZEREGOWY, PREDKOSC, timeout=0.1)
    print(f"Połączono z {PORT_SZEREGOWY}. Rozpoczynam kreślenie wykresu...")
except Exception as e:
    print(f"Błąd połączenia: {e}")
    exit()

# Bufory na dane
x_data = deque(maxlen=MAX_PUNKTOW)
y_data = deque(maxlen=MAX_PUNKTOW)
counter = 0

# Konfiguracja wykresu
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], lw=2, color='#2ecc71')

ax.set_title('Akwizycja napięcia z Arduino - Live')
ax.set_xlabel('Numer próbki')
ax.set_ylabel('Napięcie [V]')
ax.grid(True, alpha=0.3)

def update(frame):
    global counter
    
    try:
        # Odczytujemy wszystkie dostępne dane z bufora portu
        while ser.in_waiting > 0:
            raw_data = ser.readline().decode('utf-8').strip()
            if raw_data:
                # Konwersja ADC na Wolty
                adc_value = float(raw_data)
                voltage = (adc_value * V_REF) / ADC_RES
                
                x_data.append(counter)
                y_data.append(voltage)
                counter += 1
        
        if y_data:
            # Aktualizacja danych na wykresie
            line.set_data(x_data, y_data)
            
            # Dynamiczne dopasowanie osi X
            ax.set_xlim(x_data[0], x_data[-1] + 1)
            
            # Dynamiczne dopasowanie osi Y z małym marginesem
            min_y, max_y = min(y_data), max(y_data)
            margin = max(0.1, (max_y - min_y) * 0.1)
            ax.set_ylim(min_y - margin, max_y + margin)
            
    except Exception:
        pass
    return line,

# Animacja z interwałem 30ms dla zachowania płynności
ani = animation.FuncAnimation(fig, update, interval=30, blit=False)

plt.show(block=True)

# Zamknięcie portu po zamknięciu okna
ser.close()
print("Połączenie zamknięte.")