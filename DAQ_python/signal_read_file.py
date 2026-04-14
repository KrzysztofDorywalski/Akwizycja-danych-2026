import serial
import datetime
import csv
import sys

# ================= KONFIGURACJA =================
PORT_SZEREGOWY = 'COM10'  # Zmień na właściwy port
PREDKOSC = 115200

# Parametry ADC
V_REF = 5.0
ADC_RES = 1023.0

# Automatyczna nazwa pliku
NAZWA_PLIKU = f"log_napiecie_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# ================================================

try:
    ser = serial.Serial(PORT_SZEREGOWY, PREDKOSC, timeout=1)
    print(f"Połączono z {PORT_SZEREGOWY}")
    print(f"Rejestracja danych do pliku: {NAZWA_PLIKU}")
    print("Naciśnij Ctrl+C, aby zakończyć pomiar.")
except Exception as e:
    print(f"BŁĄD: Nie można otworzyć portu: {e}")
    exit()

# Przygotowanie pliku i start pętli
counter = 0
try:
    with open(NAZWA_PLIKU, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Indeks', 'Czas_Systemowy', 'Napiecie_V'])
        
        while True:
            # Odczyt linii z Arduino
            if ser.in_waiting > 0:
                try:
                    raw_data = ser.readline().decode('utf-8').strip()
                    if raw_data:
                        # 1. Konwersja na napięcie
                        adc_value = float(raw_data)
                        voltage = (adc_value * V_REF) / ADC_RES
                        
                        # 2. Znacznik czasu
                        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        
                        # 3. Zapis do pliku
                        csv_writer.writerow([counter, timestamp, f"{voltage:.4f}"])
                        
                        # 4. Opcjonalny podgląd w konsoli co 100 próbek (żeby nie zwalniać zapisu)
                        if counter % 100 == 0:
                            print(f"\rZapisano próbek: {counter} | Aktualne napięcie: {voltage:.3f}V", end="")
                        
                        counter += 1
                except (ValueError, UnicodeDecodeError):
                    # Ignoruj błędy transmisji (np. ucięte linie przy starcie)
                    continue

except KeyboardInterrupt:
    print("\n" + "="*30)
    print("PRZERWANO RĘCZNIE (Ctrl+C)")
finally:
    ser.close()
    print(f"POMIAR ZAKOŃCZONY.")
    print(f"Zapisano łącznie {counter} próbek.")
    print(f"Plik: {NAZWA_PLIKU}")
    print("="*30)