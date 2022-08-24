import time
import socket
from threading import Thread

class DataServer:
    '''
        use socket to send key data to GUI for analysis
    '''
    def __init__(self, data, ip='127.0.0.1', port=6666):
        self.ip = ip
        self.port = port
        self.data = data
        
    def data_handle(self, new_socket, addr):
        try:
            while len(self.data)>0:
                new_socket.sendall(self.data.encode('utf-8'))
                time.sleep(0.1)
        except Exception as ret:
            print(str(addr) + " error, disconnected..: " + str(ret))
        
    def run(self):
        try:
            main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            main_socket.bind((self.ip, self.port)) 
            main_socket.listen(128)  
            print("sever started...")
            while True:
                new_socket, addr = main_socket.accept()
                Thread(target=self.data_handle, args=(new_socket, addr)).start()
        except Exception as ret:
            print("server error: " + str(ret))
    

if __name__ == "__main__":
    num_robot_to_track = 4
    data = [[0, 1, 1], [1, 1, 5], [2, 5, 1], [3, 5, 5]]
    data_str = 'Detected {} of {} at 1603184275917588 \n'.format(len(data), num_robot_to_track)
    for d in data:
        data_str += 'Robot {} {:.3f} {:.3f} {:3.3f} {} \n'.format(str(d[0]).zfill(3),
                                                                d[1]/10,
                                                                d[2]/10,
                                                                101.0,
                                                                1603184275917588,
                                                                )
    data_server = DataServer(data_str)
    data_server_process = Thread(target=data_server.run)
    data_server_process.start()
    while True:
        time.sleep(1)
        # print(data_str)