import serial
import struct
import time
import csv
import threading
import queue

# ================= KONFIGURACJA =================
PORT_SZEREGOWY = 'COM10'
PREDKOSC = 115200
START_BYTE = b'\xaa'

V_REF = 5.0
ADC_RES = 1023.0

SAMPLING_INTERVAL = 0.0005   # 500 µs = 2000 Hz

PLIK = "pomiar.csv"
KOLEJKA_MAX = 10000
# ================================================

q = queue.Queue(maxsize=KOLEJKA_MAX)
running = True


def serial_reader(ser):
    global running

    while running:
        try:
            byte = ser.read(1)

            if byte == START_BYTE:
                payload = ser.read(2)

                if len(payload) < 2:
                    continue

                adc = struct.unpack('<H', payload)[0]

                if adc <= 1023:
                    voltage = adc * V_REF / ADC_RES
                    q.put(voltage)

        except Exception as e:
            print("Błąd portu:", e)
            running = False


try:
    ser = serial.Serial(PORT_SZEREGOWY, PREDKOSC, timeout=1)
    print("Otwarto port:", PORT_SZEREGOWY)

    time.sleep(2)
    ser.reset_input_buffer()

except Exception as e:
    print("Nie można otworzyć portu:", e)
    exit()


thread = threading.Thread(target=serial_reader, args=(ser,))
thread.start()

print("Rozpoczęto zapis danych. CTRL+C aby zatrzymać.")

with open(PLIK, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "voltage"])

    sample = 0

    try:
        while True:

            voltage = q.get()

            t = sample * SAMPLING_INTERVAL
            writer.writerow([t, voltage])

            sample += 1

    except KeyboardInterrupt:
        print("\nZatrzymano zapis.")
        running = False

thread.join()
ser.close()

print("Zapis zakończony.")