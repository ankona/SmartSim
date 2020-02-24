
import time
from os import environ
from redis import Redis, ConnectionError
from smartsim.error import SmartSimConnectionError

class Connection:
    """The class for SmartSimNodes to communicate with the central Redis(KeyDB)
       database to retrieve and send data from clients and other nodes"""

    def __init__(self, db_id):
        self.conn = None
        self.db_id = str(db_id)

    def connect(self, host, port):
        try:
            self.conn = Redis(host, port)
            if not self.connected():
                raise SmartSimConnectionError(
                    "Could not reach orchestrator at " + host)
        except ConnectionError as e:
            raise SmartSimConnectionError(
                "Could not reach orchestrator at " + host) from e

    def connected(self):
        """pings server and returns true if connected"""
        response = self.conn.ping()
        return response

    def get(self, key, wait=False, wait_interval=.5):
        # TODO put in a timeout limit
        prefixed_key = "_".join((self.db_id, key))
        data = self.conn.get(prefixed_key)
        if not data and wait:
            time.sleep(wait_interval)
            return self.get(key, wait=wait)
        else:
            return data

    def send(self, key, value):
        key = "_".join((self.db_id, key))
        self.conn.set(key, value)