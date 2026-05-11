import cv2

def list_ports():
    is_working = True
    dev_port = 0
    working_ports = []
    while is_working:
        cap = cv2.VideoCapture(dev_port)
        if not cap.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_working = True
            res, frame = cap.read()
            if res:
                print("Port %s is working and reads images." %dev_port)
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s ) is present but does not read." %(dev_port,cap.get(3),cap.get(4)))
        cap.release()
        dev_port +=1
    return working_ports

if __name__ == "__main__":
    ports = list_ports()
    print("Working ports:", ports)
