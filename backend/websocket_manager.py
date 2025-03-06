from fastapi import WebSocket

class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        print(f"client {websocket.client.host}: {websocket.client.port} connected")
        self.active_connections.append(websocket)

    async def disconnect(self, websocket):
        self.active_connections.remove(websocket)
        print(f"client {websocket.client.host}: {websocket.client.port} disconnected")

    async def send_message(self, websocket: WebSocket, message: str):
        messageToSend = {
            "client": f"{websocket.client.host}: {websocket.client.port}",
            "message": message
        }
        await websocket.send_json(messageToSend)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)