import asyncio
from evdev import InputDevice, list_devices, ecodes
import socket

def find_joystick():
    devices = [InputDevice(path) for path in list_devices()]
    for device in devices:
        if 'Pro Controller' in device.name or 'Gamepad' in device.name or 'Joystick' in device.name:
            print(f"? Mando detectado: {device.name} en {device.path}")
            return device
    print("? No se detectó ningún mando compatible.")
    return None

async def main():
    device = find_joystick()
    if not device:
        return

    # Create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('localhost', 1977))
    print("Connected!")

    async for event in device.async_read_loop():
        if event.type == ecodes.EV_KEY:
            name = ecodes.KEY[event.code]
            value = 'pressed' if event.value else 'released'
            msg = f"[BTN] {name} {value}"
            print(msg)
            sock.sendall((msg + '\n').encode())

        elif event.type == ecodes.EV_ABS:
            axis = ecodes.ABS[event.code]
            msg = f"[AXIS] {axis} {event.value}"
            print(msg)
            sock.sendall((msg + '\n').encode())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n? Finalize")
