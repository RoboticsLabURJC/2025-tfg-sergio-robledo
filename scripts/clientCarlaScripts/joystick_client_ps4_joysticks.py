#------------------------------------------------
# Codigo para el ordenador conectado al mando Dualshock de Play Station,
# que manda el mensaje a otro en remoto
#-------------------------------------------
#Este codigo permite mover el coche con los joysticks izquierdo y derecho, izquierdo para
# la direccion y derecho arriba y abajo para el throttle
# Es el codigo cliente para datasetgenPS4Controllerjoysticks.py
#------------------------------------------------

import socket
from evdev import InputDevice, categorize, ecodes, list_devices
import select

HOST = 'localhost'
PORT = 1977

# Buscar dispositivo de tipo joystick
devices = [InputDevice(path) for path in list_devices()]
joystick = None
for device in devices:
    if "Sony" in device.name or "Wireless Controller" in device.name or "PS4" in device.name:
        joystick = device
        break

if not joystick:
    print("No PS4 controller found")
    exit()

print(f"Using device: {joystick.path} {joystick.name}")

# Conexión
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("? Connected to receiver")

    abs_x = 0
    abs_ry = 0

    for event in joystick.read_loop():
        if event.type == ecodes.EV_ABS:
            if event.code == ecodes.ABS_X: # Joystick izquierdo (steer)
                abs_x = event.value
            elif event.code == ecodes.ABS_RY: # Joystick derecho (throttle)
                abs_ry = 255 - event.value

            # Enviar ambos valores en una línea
            msg = f"[ABS_X] {abs_x} [ABS_RY] {abs_ry}\n"
            s.sendall(msg.encode())
