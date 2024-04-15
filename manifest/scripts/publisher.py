import zmq
import uuid
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-source", type=str)
parser.add_argument("-port", type=int)
parser.add_argument("-prefix", type=str)

args = parser.parse_args()


pub_addr = f"tcp://10.150.0.3:{args.port}"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect(pub_addr)

while True:
    prefix = args.prefix
    msg = str(
        {
            "from": args.source,
            "foo": str(uuid.uuid4()),
        }
    )

    print(f"send {msg} to {pub_addr}")
    socket.send_string(f"{prefix}|{msg}")
    # time.sleep(0.1)
