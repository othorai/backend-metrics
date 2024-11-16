# connectors/base.py
from abc import ABC, abstractmethod

class BaseConnector(ABC):
    @abstractmethod
    def connect(self):
        self.source_type = None  # Will be set by child classes
        self.connection = None

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def query(self, query_string):
        pass

    @abstractmethod
    def insert(self, table, data):
        pass

    @abstractmethod
    def update(self, table, data, condition):
        pass

    @abstractmethod
    def delete(self, table, condition):
        pass

