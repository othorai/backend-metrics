from simple_salesforce import Salesforce
from app.connectors.base import BaseConnector

class SalesforceConnector(BaseConnector):
    def __init__(self, username, password, security_token, domain='login'):
        self.username = username
        self.password = password
        self.security_token = security_token
        self.domain = domain
        self.sf = None

    def connect(self):
        self.sf = Salesforce(
            username=self.username,
            password=self.password,
            security_token=self.security_token,
            domain=self.domain
        )

    def disconnect(self):
        # Salesforce doesn't require explicit disconnection
        pass

    def query(self, query_string):
        return self.sf.query_all(query_string)['records']

    def insert(self, object_name, data):
        return self.sf.__getattr__(object_name).create(data)

    def update(self, object_name, record_id, data):
        return self.sf.__getattr__(object_name).update(record_id, data)

    def delete(self, object_name, record_id):
        return self.sf.__getattr__(object_name).delete(record_id)