import socket
import struct
import sys
import threading
import time


class TCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connectionStatus = False
        self.connection = None
        self.address = None
        self.sckt = None
        self.closeConnection = False
        self.AcceptConnectionsThread = threading.Thread(target=self.AcceptConnections)
        self.GetDetectionsThread = threading.Thread(target=self.GetDetections)

    def BindSocket(self):
        try:
            self.sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sckt.setblocking(1)
            self.sckt.bind((self.host, self.port))
            self.sckt.listen(5)
        except socket.error as msg:
            print("Socket Binding error" + str(msg) + "\n" + "Retrying...")
            self.BindSocket()

    def AcceptConnections(self):
        self.BindSocket()
        while True:
            try:
                self.connection, self.address = self.sckt.accept()
                # prevents timeout
                print(
                    "Connection has been established :"
                    + str(self.address[0])
                    + "at port "
                    + str(self.address[1])
                )
                self.connectionStatus = True
            except socket.error as msg:
                self.connectionStatus = False
                print("Error accepting connection from " + str(msg))
                if self.closeConnection:
                    break
                else:
                    self.connection.close()
                    self.connection = None
                    continue

    def CloseConnection(self):
        self.closeConnection = True
        if self.connection != None:
            self.connection.close()
            self.connection = None
        if self.sckt != None:
            self.sckt.close()
            self.sckt = None
        self.AcceptConnectionsThread.join()
        self.GetDetectionsThread.join()

    def ReceiveAll(self, n):
        # Helper function to recv n bytes or return None if EOF is hit
        if self.connection == None:
            return None
        data = bytearray()
        while len(data) < n:
            packet = self.connection.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def ReceiveMessage(self):
        # Read message length and unpack it into an integer
        if self.connection == None:
            return None
        raw_msglen = self.ReceiveAll(4)
        if not raw_msglen:
            return None
        msglen = 4 * struct.unpack(">I", raw_msglen)[0]
        # Read the message data
        data = self.ReceiveAll(msglen * 4)
        return data, msglen

    def GetDetections(self):
        while not self.closeConnection:
            try:
                if self.connection == None:
                    continue
                data, n_boxes = self.ReceiveMessage()
                ll = struct.unpack("f" * n_boxes, data)
                print(ll)
            except socket.error as msg:
                print("Error receiving commands" + str(msg))
                if self.closeConnection:
                    break
                else:
                    continue

    def LaunchTCP(self):
        self.AcceptConnectionsThread.daemon = True
        self.GetDetectionsThread.daemon = True
        self.AcceptConnectionsThread.start()
        time.sleep(1)
        self.GetDetectionsThread.start()
        time.sleep(1)


def main():
    server = TCPServer(host="127.0.0.1", port=9999)
    server.LaunchTCP()
    cmd = input()
    server.CloseConnection()


if __name__ == "__main__":
    main()
