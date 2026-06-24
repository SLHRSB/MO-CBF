import socket
import struct, json
import random
import numpy as np
from datetime import datetime
import math
import os
from time import sleep
import sys
import json

pi = 3.1477

class Reciver_UDP2:
    def __init__(self, name, port_number, this_socket=None):
        self.name = name
        self.port_number = port_number
        self.this_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def build(self):
        self.this_socket.bind(("", self.port_number))
        print("waiting on port:", self.port_number)

    def get(self):
        data, addr = self.this_socket.recvfrom(64 * 4096)
        data = json.loads(data.decode("utf-8"))
        return data

    def close(self):
        self.this_socket.close()
        # print("udp closed")


class Transmitter_UDP_2:
    def __init__(self, name, port_number, host='129.97.72.1', this_socket=None):
        self.name = name
        self.port_number = port_number
        self.this_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.host = socket.gethostname()
        print("host IP:", self.host)
        print("Sending data to port:", self.port_number)

    def send_data(self, off_set, desired_velocity, throttle_flag, brake_flag, reset_flag):
        y = struct.pack("ddddd", off_set, desired_velocity, throttle_flag, brake_flag, reset_flag)
        self.this_socket.sendto(y, (self.host, self.port_number))

    def close(self):
        self.this_socket.close()


class Receiver_TCP:
    def __init__(self, name, port_number, this_socket=None):
        self.name = name
        self.port_number = port_number
        self.this_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) if not this_socket else this_socket

    def build(self):
        self.this_socket.bind(("", self.port_number))
        self.this_socket.listen(1)  # Start listening







