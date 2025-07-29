import socket

HOST = 'localhost'
PORT = 1977

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"🟢 Esperando conexión en {HOST}:{PORT}...")
    conn, addr = s.accept()
    with conn:
        print(f"🔌 Conectado desde {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            print(f"🎮 Datos recibidos: {data.decode().strip()}")
