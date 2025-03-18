import ctypes
import asyncio
import websockets
import json
import http.server
import socketserver
import threading

# C++ Structure definitions
class CaptureSettings(ctypes.Structure):
    _fields_ = [
        ("capture_width", ctypes.c_int),
        ("capture_height", ctypes.c_int),
        ("capture_x", ctypes.c_int),
        ("capture_y", ctypes.c_int),
        ("target_fps", ctypes.c_double),
        ("jpeg_quality", ctypes.c_int),
        ("paint_over_jpeg_quality", ctypes.c_int),
        ("use_paint_over_quality", ctypes.c_bool),
        ("paint_over_trigger_frames", ctypes.c_int),
        ("damage_block_threshold", ctypes.c_int),
        ("damage_block_duration", ctypes.c_int),
    ]

class StripeEncodeResult(ctypes.Structure):
    _fields_ = [
        ("stripe_y_start", ctypes.c_int),
        ("stripe_height", ctypes.c_int),
        ("size", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
    ]

StripeCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(StripeEncodeResult), ctypes.c_void_p)

# Load shared library
lib = ctypes.CDLL('./screen_capture_module.so')

# C++ function signatures
create_module = lib.create_screen_capture_module
create_module.restype = ctypes.c_void_p

destroy_module = lib.destroy_screen_capture_module
destroy_module.argtypes = [ctypes.c_void_p]

start_capture = lib.start_screen_capture
start_capture.argtypes = [ctypes.c_void_p, CaptureSettings, StripeCallback, ctypes.c_void_p]

stop_capture = lib.stop_screen_capture
stop_capture.argtypes = [ctypes.c_void_p]

free_stripe_encode_result_data = lib.free_stripe_encode_result_data
free_stripe_encode_result_data.argtypes = [ctypes.POINTER(StripeEncodeResult)]


# Globals
jpeg_queue = asyncio.Queue()
clients = set()
is_capturing = False
module = None

async def send_jpegs():
    while True:
        jpeg_bytes = await jpeg_queue.get()
        if not clients:
            jpeg_queue.task_done()
            continue
        await asyncio.gather(*(client.send(jpeg_bytes) for client in clients))
        jpeg_queue.task_done()

async def ws_handler(websocket, path):
    global is_capturing, module, capture_settings, stripe_callback, clients
    clients.add(websocket)
    print(f"Client connected: {websocket.remote_address}")

    if not is_capturing:
        print("Starting capture")
        start_capture(module, capture_settings, stripe_callback, None)
        is_capturing = True

    try:
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosedError:
        print(f"Client disconnected abruptly: {websocket.remote_address}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        clients.remove(websocket)
        print(f"Client disconnected: {websocket.remote_address}")
        if is_capturing and not clients:
            print("Stopping capture")
            stop_capture(module)
            is_capturing = False

def py_stripe_callback(result_ptr, user_data):
    global is_capturing, jpeg_queue, loop
    if is_capturing:
        result = result_ptr.contents
        if result.data:
            data = ctypes.cast(result.data,
                                     ctypes.POINTER(ctypes.c_ubyte * result.size)).contents
            asyncio.run_coroutine_threadsafe(jpeg_queue.put(bytes(data)), loop)
            free_stripe_encode_result_data(result_ptr)

# HTTP Server to serve index.html
def start_http_server():
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("localhost", 9001), Handler) as httpd:
        print("Serving index.html at localhost:9001")
        httpd.serve_forever()

if __name__ == "__main__":
    capture_settings = CaptureSettings()
    capture_settings.capture_width = 2560
    capture_settings.capture_height = 1440
    capture_settings.capture_x = 0
    capture_settings.capture_y = 0
    capture_settings.target_fps = 30.0
    capture_settings.jpeg_quality = 40
    capture_settings.paint_over_jpeg_quality = 95
    capture_settings.use_paint_over_quality = True
    capture_settings.paint_over_trigger_frames = 2
    capture_settings.damage_block_threshold = 15
    capture_settings.damage_block_duration = 30

    stripe_callback = StripeCallback(py_stripe_callback)

    module = create_module()

    if module:
        print("Capture module created.")
        try:
            # Start HTTP server in a separate thread
            http_thread = threading.Thread(target=start_http_server)
            http_thread.daemon = True 
            http_thread.start()

            start_server = websockets.serve(ws_handler, 'localhost', 9000)
            loop = asyncio.get_event_loop()
            server = loop.run_until_complete(start_server)
            send_task = loop.create_task(send_jpegs())

            print("WebSocket server started. Waiting for connections...")
            loop.run_forever()

        except KeyboardInterrupt:
            print("Stopping server...")
        finally:
            print("Cleaning up...")
            if is_capturing:
                stop_capture(module)
                print("Capture stopped.")
            if 'server' in locals() and server.is_serving():
                server.close()
                loop.run_until_complete(server.wait_closed())
            if 'send_task' in locals():
                send_task.cancel()
                try:
                    loop.run_until_complete(send_task)
                except asyncio.CancelledError:
                    pass
            destroy_module(module)
            print("Capture module destroyed.")
            loop.close()
    else:
        print("Failed to create capture module.")

