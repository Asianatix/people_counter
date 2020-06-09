import socket
import struct
import sys
import threading
import time
from datetime import datetime

TIME_OUT_IN_MICROSECONDS = 100000


class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        #        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #        self.sock.setblocking(1)
        self.sock = None
        self.connectionStatus = False
        self.boundingBoxes = []
        self.closeConnection = False
        self.mutex = threading.Lock()
        self.launchThread = threading.Thread(target=self.SendData)
        self.currentFrameNumber = 0
        self.previousFrameNumber = 0
        self.eventFlag = threading.Event()
        self.eventFlag.clear()
        self.reconnecting = False

    def ConnectToHost(self):
        if self.reconnecting:
            return
        try:
            if not self.connectionStatus:
                self.reconnecting = True
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setblocking(1)
                # print("Timeout = " + str(self.sock.gettimeout()))
                self.sock.connect((self.host, self.port))
                self.connectionStatus = True
                print("Connected")
                self.currentFrameNumber = 0
                self.previousFrameNumber = 0
                self.reconnecting = False
        except socket.error as msg:
            print("connection error:" + str(msg))
            self.connectionStatus = False
            self.reconnecting = False

    def SendData_old(self):

        while not self.closeConnection:
            self.eventFlag.wait(10)  # timeout = 10
            self.eventFlag.clear()
            if self.currentFrameNumber <= self.previousFrameNumber:
                continue
            else:
                self.previousFrameNumber = self.currentFrameNumber

            newboundingBoxes = []
            # newboundingBoxes.clear()

            self.mutex.acquire()
            if len(self.boundingBoxes) == 0:
                self.mutex.release()
                continue
            else:
                for box in self.boundingBoxes:
                    newboundingBoxes.append(box)
            self.mutex.release()

            try:
                if self.connectionStatus == True:
                    # self.sock.recv(1024)
                    if len(newboundingBoxes) > 0:
                        n_boxes = int(len(newboundingBoxes) / 4)
                        n_boxes_bytes = n_boxes.to_bytes(4, byteorder="big")
                        self.sock.sendall(n_boxes_bytes)
                        data = struct.pack(
                            "f" * len(newboundingBoxes), *newboundingBoxes
                        )
                        self.sock.sendall(data)
                else:
                    if self.closeConnection == False:
                        self.ConnectToHost()
            except socket.error as msg:
                self.sock.close()
                self.connectionStatus = False

    def SendData(self):

        while not self.closeConnection:
            self.eventFlag.wait(10)
            self.eventFlag.clear()
            if self.currentFrameNumber <= self.previousFrameNumber:
                continue
            else:
                self.previousFrameNumber = self.currentFrameNumber

            newboundingBoxes = []
            # newboundingBoxes.clear()

            self.mutex.acquire()
            if len(self.boundingBoxes) == 0:
                self.mutex.release()
                continue
            else:
                for box in self.boundingBoxes:
                    newboundingBoxes.append(box)
            self.mutex.release()

            try:
                if self.connectionStatus == True:
                    # self.sock.recv(1024)
                    if len(newboundingBoxes) > 0:
                        n_boxes = int(len(newboundingBoxes) / 4)
                        n_boxes_bytes = n_boxes.to_bytes(4, byteorder="big")
                        n_boxes_bytes_left = len(n_boxes_bytes)
                        tic = time.clock()
                        while n_boxes_bytes_left > 0 and self.connectionStatus == True:
                            n_boxes_bytes_left = len(n_boxes_bytes)
                            n_sent = self.sock.send(n_boxes_bytes)
                            if n_sent > 0:
                                n_boxes_bytes_left -= n_sent
                                if n_boxes_bytes_left > 0:
                                    n_boxes_bytes = n_boxes_bytes[n_sent:]
                            else:
                                toc = time.clock()
                                if (toc - tic) > 3:
                                    self.connectionStatus = False

                        # self.sock.sendall(n_boxes_bytes)
                        data = struct.pack(
                            "f" * len(newboundingBoxes), *newboundingBoxes
                        )
                        # self.sock.sendall(data)
                        data_bytes_left = len(data)
                        while data_bytes_left > 0 and self.connectionStatus == True:
                            data_bytes_left = len(data)
                            n_sent = self.sock.send(data)
                            data_bytes_left -= n_sent
                            if n_sent > 0:
                                if data_bytes_left > 0:
                                    data = data[n_sent:]
                                else:
                                    toc = time.clock()
                                    if (toc - tic) > 3:
                                        self.connectionStatus = False

                else:
                    if self.closeConnection == False:
                        self.ConnectToHost()
            except socket.error as msg:
                self.sock.close()
                self.connectionStatus = False

    def SendBoundingBoxes(self, boundingBoxes):
        self.mutex.acquire()
        self.boundingBoxes.clear()
        self.boundingBoxes = boundingBoxes.copy()
        self.currentFrameNumber = self.currentFrameNumber + 1
        self.mutex.release()
        self.eventFlag.set()

    def CloseConnection(self):
        self.closeConnection = True
        self.launchThread.join()

    def LaunchConnection(self):
        self.launchThread.daemon = True
        self.launchThread.start()
        time.sleep(1)


def main():
    # client = TCPClient('127.0.0.1',9999)
    client = TCPClient("192.168.50.1", 9999)
    client.LaunchConnection()
    counter = 0
    boundingBoxesList = []
    boundingBoxes1 = [1.0, 2.5, 6.3, 8.9]
    boundingBoxesList.append(boundingBoxes1)
    boundingBoxes2 = [
        3.0,
        7.1,
        7.5,
        1.2,
        3.0,
        7.1,
        7.5,
        1.2,
        3.0,
        7.1,
        7.5,
        1.2,
        3.0,
        7.1,
        7.5,
        1.2,
    ]
    boundingBoxesList.append(boundingBoxes2)
    boundingBoxes3 = [6.3, 2.8, 2.3, 9.7, 7.8, 6.4, 7.8, 6.4]
    boundingBoxesList.append(boundingBoxes3)
    boundingBoxes4 = [7.4, 4.2, 3.1, 8.1, 7.4, 4.2, 3.1, 8.1, 7.4, 4.2, 3.1, 8.1]
    boundingBoxesList.append(boundingBoxes4)
    while counter < 200000:
        client.SendBoundingBoxes(boundingBoxesList[counter % 4])
        counter += 1
        print(counter)
        time.sleep(0.05)
    client.CloseConnection()


if __name__ == "__main__":
    main()
